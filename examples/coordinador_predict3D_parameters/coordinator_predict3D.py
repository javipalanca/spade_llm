
import asyncio
import json
import logging
import os
import random
from datetime import datetime


import joblib
import pandas as pd
from colorama import Fore, Style
import spade
from spade.agent import Agent
from spade.behaviour import CyclicBehaviour
from spade.message import Message
from spade_llm.agent import ChatAgent
from spade_llm.agent.coordinator_agent import CoordinatorAgent
from spade_llm.providers import LLMProvider
from spade_llm.tools import LLMTool  
from spade_llm.utils import load_env_vars

# === BASIC CONFIGURATION ===
MODEL_PATH = os.path.join("models", "regresion3d_simple.joblib")
# For the demo: on the first prediction we force a validation error by simulating nozzle_temperature=0
SIMULATE_VALIDATION_ERRORS = True

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logging.getLogger("spade_llm").setLevel(logging.INFO)

# === TUNER: agent that predicts parameters with a model ===
class TunerAgent(Agent):
    class PredictTuningBehaviour(CyclicBehaviour):

        def predict(self, hardness: float, material: str):
            """Returns a dict with model predictions for the target columns.

            Inputs:
            - hardness: numeric value (e.g., 50, 60...).
            - material: label known by the model one-hot (e.g., "PLA", "ABS").
            """
            model = self.agent.bundle["model"]
            cols = self.agent.bundle["feature_columns"]
            hardness_col = self.agent.bundle["hardness_col"]          # "tension_strenght"
            mat_prefix = self.agent.bundle["material_prefix"]         # "material_"
            targets = self.agent.bundle["targets"]

            # Build input vector with the exact training columns
            X = pd.DataFrame([[0] * len(cols)], columns=cols)
            X.loc[0, hardness_col] = hardness

            mat_col = f"{mat_prefix}{material}"
            if mat_col in X.columns:
                X.loc[0, mat_col] = 1
            # If the material did not exist in training, keep all one-hots at 0

            y_pred = model.predict(X)[0]
            return {targets[i]: float(y_pred[i]) for i in range(len(targets))}
        
        async def run(self):
            """Waits for a JSON message {"hardness": num, "material": str} and responds
            with a JSON array: [speed, layer_height, extruder_temperature, bed_temperature]."""
            msg = await self.receive(timeout=30)
            if not msg:
                return

            # Try to read JSON with {"hardness": int/float, "material": str}
            try:
                data = json.loads(msg.body) if msg.body else {}
                _hardness = data.get("hardness")
                _material = data.get("material")
            except Exception:
                _hardness = None
                _material = None

            pred_value = self.predict(_hardness, _material)
            print(Fore.GREEN + f"PREDICTOR AGENT: pred_value={pred_value}" + Style.RESET_ALL)
            speed = int(pred_value["print_speed"])
            layer_height = round(pred_value["layer_height"], 2)
            bed_temperature = int(pred_value["bed_temperature"])
            extruder_temperature = int(pred_value["nozzle_temperature"])

            # For the demo: first time we force nozzle_temperature=0 so
            # the validator reports an error. On the second attempt it will pass.
            if self.agent.run_count == 0 and SIMULATE_VALIDATION_ERRORS:
                values = [speed, layer_height, 0, bed_temperature]
            else:
                values = [speed, layer_height, extruder_temperature, bed_temperature]
            self.agent.run_count += 1
            reply = msg.make_reply()
            # Return ONLY the array of 4 numbers as JSON (user requirement)
            reply.body = json.dumps(values, ensure_ascii=False)
            await self.send(reply)

    async def setup(self):
        """Load the model bundle and prepare the cyclic behaviour."""
        self.bundle = joblib.load(MODEL_PATH)
        self.run_count = 0
        self.add_behaviour(self.PredictTuningBehaviour())

# === VALIDATOR: agent that validates plausible parameter ranges ===
class ValidatorAgent(Agent):
    class ValidateBehaviour(CyclicBehaviour):
        async def run(self):
            """Receives [v, h, te, tc] or an object with those fields and returns {ok, reasons?, expected?}."""
            msg = await self.receive(timeout=30)
            if not msg:
                return
            try:
                data = json.loads(msg.body) if msg.body else {}
                # Support array [v, h, te, tc] or object with fields
                if isinstance(data, list) and len(data) == 4:
                    speed, layer_height, extruder_temperature, bed_temperature = data
                else:
                    speed = data.get("speed")
                    layer_height = data.get("layer_height")
                    extruder_temperature = data.get("extruder_temperature")
                    bed_temperature = data.get("bed_temperature")
                errors = []

                # Plausible ranges
                if not (30 <= float(speed) <= 120): errors.append("speed out of range [30,120] mm/s")
                if not (0.10 <= float(layer_height) <= 0.30): errors.append("layer_height out of range [0.10,0.30] mm")
                if not (180 <= float(extruder_temperature) <= 240): errors.append("extruder_temperature out of range [180,240] °C")
                if not (50 <= float(bed_temperature) <= 70): errors.append("bed_temperature out of range [50,70] °C")

                result = {"ok": len(errors) == 0}
                if errors:
                    result["reasons"] = errors
                    result["expected"] = {
                        "speed": "[30,120] mm/s",
                        "layer_height": "[0.10,0.30] mm",
                        "extruder_temperature": "[180,240] °C",
                        "bed_temperature": "[50,70] °C",
                    }
                    print(Fore.RED + f"VALIDATOR AGENT: {errors}" + Style.RESET_ALL)
                else:
                    print(Fore.GREEN + "VALIDATOR AGENT: Valid parameters" + Style.RESET_ALL)
                reply = msg.make_reply()
                reply.body = json.dumps(result, ensure_ascii=False)
                await self.send(reply)
            except Exception as e:
                reply = msg.make_reply()
                reply.body = json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False)
                await self.send(reply)

    async def setup(self):
        """Start the validator behaviour."""
        self.add_behaviour(self.ValidateBehaviour())

# === TOOL: Tool-like function to save the result to disk ===
RESULTS_SAVE_PATH = os.path.join(os.path.dirname(__file__), f"tuning_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

def create_save_settings_tool(path: str) -> LLMTool:
    """Creates a tool invokable by the coordinator to save parameters to JSON."""
    def save_settings(hardness: float, material: str, speed: float, layer_height: float, extruder_temperature: float, bed_temperature: float) -> str:
        """Saves the 6 parameters to a JSON in 'path' and returns a short message."""
        payload = {
            "hardness": hardness,
            "material": material,
            "speed": speed,
            "layer_height": layer_height,
            "extruder_temperature": extruder_temperature,
            "bed_temperature": bed_temperature,
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(Fore.GREEN + Style.DIM + f"TOOL-SAVE: Parameters saved to {path}" + Style.RESET_ALL)
        return f"Saved to {path}"
    return LLMTool(
        name="save_settings",
        description="Saves the 6 printing parameters (hardness, material, speed, layer_height, extruder_temperature, bed_temperature) to a JSON file",
        parameters={
            "type": "object",
            "properties": {
                "hardness": {"type": "number"},
                "material": {"type": "string"},
                "speed": {"type": "number"},
                "layer_height": {"type": "number"},
                "extruder_temperature": {"type": "number"},
                "bed_temperature": {"type": "number"},
            },
            "required": ["hardness", "material", "speed", "layer_height", "extruder_temperature", "bed_temperature"],
        },
        func=save_settings,
    )

# === Coordinator PROMPT (strict orchestration instructions) ===
COORDINATOR_PROMPT = """You are a strict coordinator for 3D printing parameters.
Mandatory sequence:
1) Call EXACTLY once send_to_agent with agent_id "tuner@localhost" sending JSON {"hardness": <number>, "material": "<text>"}.
2) With the tuner response (array [speed, layer_height, extruder_temperature, bed_temperature]), call send_to_agent with agent_id "validator@localhost" sending an object with those 4 fields.
3) If validator responds {"ok": true}, call the save_settings tool with the 6 fields (material, hardness, layer height, extruder temperature, bed temperature, and speed) and then reply to the user with the save message. Finish with <TASK_COMPLETE>.
4) If validator responds {"ok": false}, you may try ONE second call to the tuner adjusting "hardness" by ±10 as appropriate, validate again and if valid, save and finish. If still invalid, report the reasons and finish with <TASK_COMPLETE>.
Do not repeat steps outside of the above nor call more tools than indicated.
"""

async def main():
    """Demo entry point: configure LLM/provider, create agents and execute the flow."""
    print(Fore.MAGENTA + Style.BRIGHT + "=" * 60)
    print("TUNER COORDINATOR EXAMPLE - SPADE basic tool agent")
    print("=" * 60)
    print(Style.RESET_ALL)

    # 1) Load environment variables (e.g., OLLAMA_BASE_URL, OLLAMA_MODEL)
    load_env_vars()

    # Configure Ollama from env vars or defaults
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "https://localhost:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

    print(Fore.CYAN + Style.BRIGHT + "🔧 Configuration:")
    print("  • SPADE server: localhost:5222 (embedded)")
    print(f"  • Provider: Ollama ({OLLAMA_MODEL}) at {OLLAMA_BASE_URL}")
    print(Style.RESET_ALL)

    # 2) LLM Provider (low temperature to follow instructions precisely)
    provider = LLMProvider.create_ollama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.1,
        timeout=120.0,
    )

    print(Fore.GREEN + Style.BRIGHT + "🤖 Creating agents...")

    # 3) Create agents: tuner, validator, coordinator and chat (demo client)
    tuner = TunerAgent(
        jid="tuner@localhost",
        password="tune_pass",
    )
    print("  ✓ Tuner agent created")

    # Instantiate validator
    validator = ValidatorAgent(
        jid="validator@localhost",
        password="val_pass",
    )
    print("  ✓ Validator agent created")

    coordinator = CoordinatorAgent(
        jid="coordinator@localhost",
        password="coord_pass",
        subagent_ids=[
            "tuner@localhost",
            "validator@localhost",
        ],
        coordination_session="tuner_session",
        provider=provider,
        system_prompt=COORDINATOR_PROMPT,
        tools=[create_save_settings_tool(RESULTS_SAVE_PATH)], 
        verify_security=False,
    )
    print("  ✓ Coordinator created (manages: tuner@localhost, validator@localhost)")

    completion_detected = asyncio.Event()

    def display_callback(message: str, sender: str):
        # Prints coordinator/chat messages and marks completion if it sees <TASK_COMPLETE>
        print(f"📩 Reply from {sender}:")
        print(f"   {message}\n")
        if "<TASK_COMPLETE>" in message or "<END>" in message or "<DONE>" in message:
            completion_detected.set()
        print(Style.RESET_ALL)    

    chat_agent = ChatAgent(
        jid="user@localhost",
        password="user_pass",
        target_agent_jid="coordinator@localhost",
        display_callback=display_callback,
        verify_security=False,
    )
    print("  ✓ Chat agent created")
    print()

    print("🚀 Starting agents...")
    try:
        # 4) Start all agents
        await tuner.start()
        print("  ✓ Tuner started")
        await validator.start()  
        print("  ✓ Validator started")
        await coordinator.start()
        print("  ✓ Coordinator started")
        await chat_agent.start()
        print("  ✓ Chat agent started")

        print("\n⏳ Waiting for connections...")
        await asyncio.sleep(2)
        print("✅ All agents ready!\n" + Style.RESET_ALL)

        print(Fore.MAGENTA + Style.BRIGHT + "=" * 60)
        print("SCENARIO: Send hardness/material to Tuner and return 4 parameters")
        print("=" * 60)
        print(Style.RESET_ALL)

        # Test parameters
        hardness_string = random.choice(["low", "medium", "high"])
        material = random.choice(["PLA", "ABS"])
        print(f"🔎 Test parameters: hardness={hardness_string}, material={material}\n")
        # Test request: now the coordinator follows the sequence (tuner -> validator -> save)
        test_request = f"""I need to tune 3D printing parameters for a part that must have {hardness_string} hardness and is made of {material}.
        Please generate suitable parameters, validate that they are correct, and save them to a JSON file."""

        print("📤 Sending request to coordinator...\n")
        chat_agent.send_message(test_request)

        await asyncio.sleep(1)

        print("⏳ Waiting for coordination to complete (max 60s)...\n")
        try:
            await asyncio.wait_for(completion_detected.wait(), timeout=60)
            print(Fore.MAGENTA + Style.BRIGHT + "\n============================================================")
            print("COORDINATION COMPLETED SUCCESSFULLY")
            print("============================================================\n" + Style.RESET_ALL)
        except asyncio.TimeoutError:
            print("\n⚠️  Timeout reached. Check that:")
            print("   • The SPADE server is running (spade run)")
            print("   • Ollama is accessible and the model is available")
            print("   • The network is operational")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n🛑 Stopping agents...")
        await chat_agent.stop();     print("  ✓ Chat agent stopped")
        await coordinator.stop();    print("  ✓ Coordinator stopped")
        await validator.stop();      print("  ✓ Validator stopped")  
        await tuner.stop();          print("  ✓ Tuner stopped")
        print(Fore.MAGENTA + Style.BRIGHT + "\n============================================================")
        print("EXAMPLE FINISHED")
        print("============================================================" + Style.RESET_ALL)


if __name__ == "__main__":
    print()
    print("🔍 Pre-flight checks:")
    print("  • SPADE server running? (spade run)")
    print("  • Ollama accessible and model loaded? (OLLAMA_BASE_URL / OLLAMA_MODEL)")
    print()
    print("Press Ctrl+C to cancel, or wait to continue...\n")

    try:
        spade.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Cancelled by user")
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
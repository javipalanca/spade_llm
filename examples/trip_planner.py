"""
Valencia Trip Planner - Multi-Agent Workflow Example

Multi-agent workflow for comprehensive Valencia trip planning:
Airbnb Search → Route Planning → Plan Creation → Price Review → Final Plan

PREREQUISITES:
1. Start SPADE built-in server in another terminal:
   spade run
   
   (Advanced server configuration available but not needed)

2. Install dependencies:
   pip install spade_llm

This example uses SPADE's default built-in server (localhost:5222) - no account registration needed!

Uses:
- Airbnb MCP for accommodation search
- Valencia Smart City MCP for real-time city data
- Multi-agent coordination with conditional routing
"""

import asyncio
import os
from datetime import datetime
import spade
import logging

from spade_llm.agent import LLMAgent, ChatAgent
from spade_llm.routing import RoutingResponse
from spade_llm.providers import LLMProvider
from spade_llm.mcp import StdioServerConfig
from spade_llm.utils import load_env_vars

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logging.getLogger("spade_llm").setLevel(logging.INFO)


def price_routing_function(msg, response, context):
    """Routes price reviewer decisions based on budget analysis."""
    domain = str(msg.sender).split('@')[1]
    response_lower = response.lower()

    if "<plan_approved>" in response_lower:
        return RoutingResponse(
            recipients=f"output@{domain}",
            transform=lambda x: x.replace("<PLAN_APPROVED>", "").strip(),
            metadata={"status": "approved", "workflow": "completed"},
        )
    elif "<expensive_plan>" in response_lower:
        return RoutingResponse(
            recipients=f"airbnb@{domain}",
            transform=lambda x: f"BUDGET REVISION REQUEST:\n{x.replace('<EXPENSIVE_PLAN>', '').strip()}",
            metadata={"revision_type": "budget_optimization"},
        )
    elif "<revision_needed>" in response_lower:
        return RoutingResponse(
            recipients=f"routeplanner@{domain}",
            transform=lambda x: f"PLAN REVISION REQUEST:\n{x.replace('<REVISION_NEEDED>', '').strip()}",
        )
    else:
        # Default to plan maker for minor adjustments
        return RoutingResponse(recipients=f"routeplanner@{domain}")


async def main():
    print("=== Valencia Trip Planner ===")

    load_env_vars()
    model = os.environ.get("LLM_MODEL")
    if not model:
        raise SystemExit("LLM_MODEL is not set — copy .env.example to .env and configure it.")
    XMPP_SERVER = os.environ.get("XMPP_SERVER", "localhost")

    agents_config = {
        "airbnb": (f"airbnb@{XMPP_SERVER}", "Airbnb Search Agent"),
        "routeplanner": (f"routeplanner@{XMPP_SERVER}", "Route Planner Agent"),
        "pricereviewer": (f"pricereviewer@{XMPP_SERVER}", "Price Reviewer Agent"),
        "output": (f"output@{XMPP_SERVER}", "Output Agent"),
        "human": (f"human@{XMPP_SERVER}", "Human Agent"),
    }
    passwords = {role: f"{role}_pass" for role in agents_config}

    # Create LLM provider
    provider = LLMProvider(
        model=model,
        temperature=0.7,
    )

    # MCP Server configurations
    local_mcp_path = os.environ.get("VALENCIA_MCP_PATH")
    if local_mcp_path:
        valencia_mcp = StdioServerConfig(
            name="ValenciaSmart",
            command="uv",
            args=["run", local_mcp_path],
            cache_tools=True,
        )
    else:
        # Fetch and run directly from GitHub (requires uv)
        valencia_mcp = StdioServerConfig(
            name="ValenciaSmart",
            command="uv",
            args=[
                "run",
                "https://raw.githubusercontent.com/olafmeneses/SmartCityMCP/refs/heads/master/valencia_smart_city_mcp.py",
            ],
            cache_tools=True,
        )

    # Airbnb MCP
    airbnb_mcp = StdioServerConfig(
        name="AirbnbSearch",
        command="npx",
        args=["-y", "@openbnb/mcp-server-airbnb", "--ignore-robots-txt"],
        cache_tools=True
    )

    # Create agents dictionary
    agents = {}

    # 1. Airbnb Search Agent
    print("Creating agents...")
    agents["airbnb"] = LLMAgent(
        jid=agents_config["airbnb"][0],
        password=passwords["airbnb"],
        provider=provider,
        reply_to=agents_config["routeplanner"][0],
        system_prompt="""
                    You are an Airbnb accommodation specialist for Valencia, Spain.
                    
                    CRITICAL: You must select ONE best apartment and include ALL detailed information in your response, as the next agent needs this complete data.
                    
                    Your task:
                    1. Search for Airbnb accommodations based on user requirements
                    2. Analyze multiple options
                    3. Select THE BEST SINGLE apartment based on location, price, and value
                    4. Include COMPLETE details in your response
                    
                    RESPONSE FORMAT (MANDATORY):
                    === SELECTED VALENCIA ACCOMMODATION ===
                    CHOSEN APARTMENT:
                    
                    Name: [exact name]
                    Location: [full address/area with neighborhood]
                    Price: €[amount] per night
                    Total Cost: €[price × nights] for [X] nights
                    Rating: [rating]/5 ([number] reviews)
                    Capacity: [number] guests
                    Bedrooms: [number]
                    Bathrooms: [number]
                    Amenities: [complete list including WiFi, kitchen, AC, etc.]
                    Host: [host name and rating if available]
                    Neighborhood: [area description - historic center, beach area, etc.]
                    URL: [if available]
                    
                    SELECTION REASONING:
                    
                    Why chosen: [explain why this is the optimal choice]
                    Location advantages: [transport links, attractions, safety]
                    Value assessment: [price vs amenities vs location]
                    
                    === LOCATION CONTEXT FOR ROUTE PLANNING ===
                    
                    Exact address/area: [for route optimization]
                    
                    === NEXT STEPS FOR ROUTE PLANNER ===
                    The route planner should consider:
                    
                    Weather conditions for activity planning
                    Bike availability (ValenBici) near this specific location
                    Optimal routes from this accommodation to main attractions
                    Public transport options from this exact location
                                        
                    Include EVERY detail. The route planner needs complete location information to create optimal plans.""",
        mcp_servers=[airbnb_mcp]
    )

    # 2. Route Planner Agent
    agents["routeplanner"] = LLMAgent(
        jid=agents_config["routeplanner"][0],
        password=passwords["routeplanner"],
        provider=provider,
        reply_to=agents_config["pricereviewer"][0],
        system_prompt="""You are a Valencia bike route specialist with access to real-time city data focused on cycling experiences.

                        CRITICAL: You will receive apartment location data. Use this to create a BIKE-FOCUSED Valencia experience.
                        
                        Your task:
                        1. Extract the exact apartment address/location from the previous agent
                        2. Check weather conditions for the planned dates using Valencia Smart City tools
                        3. IF WEATHER IS GOOD for cycling:
                        - Find ValenBici stations near key Valencia attractions (Plaza Ayuntamiento, City of Arts & Sciences, Central Market, Cathedral, etc.)
                        - Check bike availability at these stations
                        - Check air quality conditions for cycling
                        4. Create a comprehensive BIKE ROUTE from the apartment through different Valencia zones
                        5. IMPORTANT: You only need bikes available at ONE station (near the apartment)
                        
                        RESPONSE FORMAT (MANDATORY):
                        === VALENCIA BIKE ROUTE PLAN ===
                        APARTMENT LOCATION:
                        
                        Location: [full address/area with neighborhood]
                        Price: €[amount] per night
                        Total Cost: €[price × nights] for [X] nights
                        Rating: [rating]/5 ([number] reviews)
                        Capacity: [number] guests
                        Bedrooms: [number]
                        Bathrooms: [number]
                        Amenities: [complete list including WiFi, kitchen, AC, etc.]
                        Host: [host name and rating if available]
                        Neighborhood: [area description - historic center, beach area, etc.]
                        URL: [if available] 
                        Starting point for bike route: [precise location]
                        
                        WEATHER ANALYSIS:
                        
                        Current conditions: [temperature, precipitation, wind]
                        Forecast for trip dates: [day-by-day weather]
                        Cycling suitability: [GOOD/POOR for biking with reasoning]
                        
                        AIR QUALITY CHECK:
                        
                        Current air quality index: [number and description]
                        Pollution levels: [safe/moderate/unhealthy for cycling]
                        Best cycling hours: [morning/afternoon/evening recommendations]
                        
                        VALENBICI STATION ANALYSIS:
                        Starting Station (Near Apartment):
                        
                        Station name: [closest to apartment]
                        Distance from apartment: [walking time]
                        Bikes available: [current count]
                        Station status: [operational/maintenance]
                        
                        Key Attractions with ValenBici Access:
                        
                        Plaza Ayuntamiento: Station [name], [X] bikes available
                        City of Arts & Sciences: Station [name], [X] bikes available
                        Central Market: Station [name], [X] bikes available
                        Cathedral/Historic Center: Station [name], [X] bikes available
                        Beach (Malvarossa): Station [name], [X] bikes available
                        
                        PROPOSED BIKE ROUTE:
                        🚴‍♂️ VALENCIA CYCLING TOUR - [X] KM TOTAL
                        STARTING POINT: Apartment → [X] min walk to [Station Name]
                        Pick up bike at: [Station name with current availability]
                        ROUTE ZONES:
                        Zone 1: Historic Valencia (2-3 hours)
                        
                        Cathedral & Miguelete Tower
                        Central Market
                        Silk Exchange (La Lonja)
                        Plaza Ayuntamiento
                        Route: [specific streets and bike paths]
                        Distance: [X] km | Estimated time: [X] hours
                        
                        Zone 2: Modern Valencia (2-3 hours)
                        
                        City of Arts & Sciences
                        Oceanogràfic area
                        Turia Gardens bike path
                        Route: [specific bike paths and streets]
                        Distance: [X] km | Estimated time: [X] hours
                        
                        Zone 3: Beach & Seafront (2-3 hours)
                        
                        Malvarossa Beach
                        Marina Real Juan Carlos I
                        Seaside promenade
                        Route: [coastal bike paths]
                        Distance: [X] km | Estimated time: [X] hours
                        
                        CYCLING CONDITIONS SUMMARY:
                        
                        Weather suitability: [GOOD/FAIR/POOR with reasoning]
                        Air quality: [SAFE/MODERATE/POOR for cycling]
                        Bike availability: [CONFIRMED at starting station]
                        Route difficulty: [EASY/MODERATE/CHALLENGING]
                        Total cycling distance: [X] km
                        Estimated total time: [X] hours (including stops)
                        
                        ALTERNATIVE PLAN (if weather poor):
                        [Brief indoor alternatives if cycling not recommended]
                        ROUTE_PLANNING_COMPLETE
                        
                        Focus on creating the perfect bike experience using real Valencia data.""",
        mcp_servers=[valencia_mcp],
    )


    # 4. Price Reviewer Agent
    agents["pricereviewer"] = LLMAgent(
        jid=agents_config["pricereviewer"][0],
        password=passwords["pricereviewer"],
        provider=provider,
        routing_function=price_routing_function,
        system_prompt="""You are a Valencia trip plan reviewer and quality controller.

                   CRITICAL: You will receive apartment data AND bike route plan. Your job is to review everything and make routing decisions.

                   Your task:
                   1. Review the apartment selection for value and pricing
                   2. Review the bike route plan for feasibility and safety
                   3. Make one of three routing decisions based on your analysis

                   EVALUATION CRITERIA:
                   - Apartment: Is the price reasonable for Valencia market? Good location for the planned activities?
                   - Route: Is the bike route practical? Weather suitable? Bikes actually available? Air quality safe?

                   RESPONSE FORMAT (MANDATORY):
                   === VALENCIA TRIP PLAN REVIEW ===
                   #APARTMENT EVALUATION:

                   Name: [from previous agent]
                   Price: €[amount] per night
                   Market assessment: [FAIR/EXPENSIVE/CHEAP for Valencia]
                   Location for bike route: [EXCELLENT/GOOD/POOR accessibility]
                   Value verdict: [APPROVED/OVERPRICED]

                   ##BIKE ROUTE EVALUATION:

                   Weather conditions: [SUITABLE/MARGINAL/UNSUITABLE for cycling]
                   Air quality: [SAFE/MODERATE/POOR for cycling]
                   Bike availability: [CONFIRMED/UNCERTAIN/UNAVAILABLE]
                   Route safety: [SAFE/MODERATE/RISKY]
                   Route feasibility: [PRACTICAL/CHALLENGING/UNREALISTIC]
                   Overall route verdict: [APPROVED/NEEDS_REVISION]

                   # FINAL DECISION:
                   [Choose ONE of the following]
                   1 option : 
                   ✅ PLAN APPROVED - SEND TO PUBLISHER:
                   Both apartment and bike route are excellent. Ready for final markdown publication.
                   VALENCIA TRIP PLAN - FINAL VERSION
                   🏠 Your Valencia Apartment

                   Name: [apartment name]
                   Location: [full address and area]
                   link : [URL]
                   Price: €[amount] per night (€[total] for [X] nights)
                   Rating: [rating]/5 ([reviews] reviews)
                   Amenities: [key amenities list]
                   Why Perfect: [location advantages for biking]

                   🚴‍♂️ ## Valencia Bike Route Experience

                   Weather: [conditions] - Perfect for cycling!
                   Air Quality: [status] - Safe for outdoor activities
                   Starting Point: [ValenBici station near apartment]
                   Bikes Available: ✅ [number] bikes at pickup station

                   Route Highlights:
                   Zone 1: Historic Valencia ([X] km)

                   Cathedral & Miguelete Tower
                   Central Market & Silk Exchange
                   Plaza Ayuntamiento

                   Zone 2: Modern Valencia ([X] km)

                   City of Arts & Sciences
                   Turia Gardens bike path
                   Modern architecture tour

                   Zone 3: Beach & Coast ([X] km)

                   Malvarossa Beach
                   Marina & seafront promenade
                   Seaside cycling paths

                   Cycling Conditions:

                   Total Distance: [X] km
                   Estimated Time: [X] hours with sightseeing stops
                   Difficulty: [level] - suitable for tourists
                   Best Times: [recommended hours based on weather/air quality]
                   Safety: Dedicated bike lanes for [X]% of route

                   💰 Budget Summary

                   Accommodation: €[total]
                   Bike Rental: €[amount] (ValenBici day pass)
                   Total: €[total] for [X] days

                   <PLAN_APPROVED>

                   2 option : 
                   🏠 APARTMENT TOO EXPENSIVE - NEEDS CHEAPER OPTION:
                   The apartment at €[amount]/night exceeds reasonable Valencia pricing. Need alternative accommodation.
                   <EXPENSIVE_PLAN>

                    3 option : 
                   🚴‍♂️ BIKE ROUTE NEEDS REVISION:
                   Issues with the bike route: [specific problems with weather/air quality/bike availability/safety]. Route planner needs to revise.
                   <REVISION_NEEDED>
                   DETAILED REASONING:
                   [Explain your decision with specific issues and recommendations]

                   Make decisive routing decisions to ensure the final plan is both practical and excellent value."""
    ,

    termination_markers=["<PLAN_APPROVED>"]
    )

    # 5. Output Agent (for final plan storage)

    class OutputAgent(spade.agent.Agent):
        async def setup(self):
            class OutputBehaviour(spade.behaviour.CyclicBehaviour):
                async def run(self):
                    msg = await self.receive(timeout=1.0)
                    if msg:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"valencia_trip_plan_{timestamp}.txt"

                        with open(filename, 'w', encoding='utf-8') as f:
                            f.write("VALENCIA TRIP PLAN\n")
                            f.write("=" * 50 + "\n\n")
                            f.write(msg.body)
                            f.write(f"\n\nGenerated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

                        print(f"\nTrip plan saved to: {filename}")
                        print("-" * 50)
                        print(msg.body)
                        print("-" * 50)

                    await asyncio.sleep(0.1)

            self.add_behaviour(OutputBehaviour())

    agents["output"] = OutputAgent(
        agents_config["output"][0],
        passwords["output"]
    )

    # 6. Human Agent (ChatAgent for user interaction)
    def display_callback(message, sender):
        agent_name = sender.split('@')[0].upper()
        print(f"\n[{agent_name}]")
        print(message)
        if "VALENCIA TRIP PLAN APPROVED" in message:
            print("\nWorkflow completed. Trip plan saved.")

    agents["human"] = ChatAgent(
        jid=agents_config["human"][0],
        password=passwords["human"],
        target_agent_jid=agents_config["airbnb"][0],
        display_callback=display_callback
    )

    # Start all agents
    print("Starting agents...")
    for name, agent in agents.items():
        await agent.start()

    print("All agents ready.")
    print("\nDescribe your Valencia trip (duration, people, budget, interests).")
    print("Type 'exit' to quit.\n")

    await agents["human"].run_interactive(
        input_prompt="Trip> ",
        exit_command="exit",
        response_timeout=60.0,
    )

    for agent in agents.values():
        await agent.stop()

    print("Done.")


if __name__ == "__main__":
    spade.run(main())
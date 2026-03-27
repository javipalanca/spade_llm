# Contributing

We welcome contributions to SPADE_LLM! This guide will help you get started.

## How to contribute

1. **Fork the repository** on GitHub
2. **Clone your fork** locally
3. **Create a branch** for your changes
4. **Make your changes** and test them
5. **Submit a pull request**

## Development Setup

See the [Development Guide](development.md) for detailed instructions on setting up your environment, running tests, and contributing code.

## Types of Contributions

### Bug Reports

When reporting bugs, please include:

- **Clear description** of the problem
- **Steps to reproduce** the issue
- **Expected vs actual behavior**
- **Environment details** (Python version, OS, etc.)
- **Minimal code example** that reproduces the bug

### Feature Requests

For new features, please:

- **Check existing issues** to avoid duplicates
- **Describe the use case** and motivation
- **Propose implementation approach** if possible
- **Consider backward compatibility**

## Submitting Changes

### Pull Request Process

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write code
   - Add tests
   - Update documentation

3. **Test your changes**
   ```bash
   uv run pytest
   uv run ruff check .
   ```

4. **Commit with clear messages**
   ```bash
   git add .
   git commit -m "Add feature: clear description of changes"
   ```

5. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Create pull request** on GitHub

### Pull Request Guidelines

- **Clear title** describing the change
- **Detailed description** of what and why
- **Link related issues** using keywords (fixes #123)
- **Include screenshots** for UI changes
- **Check that CI passes**

## Development Guidelines

### Architecture Principles

- **Extend, don't replace** SPADE functionality
- **Maintain compatibility** with existing SPADE agents
- **Use async/await** throughout
- **Keep interfaces simple** and consistent
- **Favor composition** over inheritance


## Release Process

### Version Numbering

We use semantic versioning (semver):

- **Major** (X.0.0) - Breaking changes
- **Minor** (0.X.0) - New features, backward compatible
- **Patch** (0.0.X) - Bug fixes, backward compatible

### Release Checklist

- [ ] All tests pass
- [ ] Documentation updated
- [ ] Changelog updated
- [ ] Version bumped
- [ ] Release notes prepared
- [ ] Tagged release created

## Community

### Getting Help

- **GitHub Issues** - Bug reports and feature requests
- **Discussions** - General questions and ideas
- **Documentation** - Comprehensive guides and API reference

### Code of Conduct

- **Be respectful** and inclusive
- **Help others** learn and contribute
- **Focus on constructive** feedback
- **Follow project** guidelines and standards

## Recognition

Contributors are recognized in:

- **CONTRIBUTORS.md** file
- **Release notes** for significant contributions
- **Documentation** acknowledgments

Thank you for contributing to SPADE_LLM!

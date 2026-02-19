# Contributing to Dragonchess

## Development Guidelines

This project maintains strict standards for code quality, documentation, and testing. Contributions should adhere to the following guidelines.

### Code Style

The codebase follows standard C++17 conventions with the following specifics:

- Four-space indentation (no tabs)
- Opening braces on the same line for functions and control structures
- Descriptive variable names following snake_case for locals and member variables
- Class names using PascalCase
- Constants and macros in UPPER_SNAKE_CASE
- Comprehensive inline documentation for complex algorithms

### Testing Requirements

All contributions that modify game logic or move generation must include corresponding unit tests. The existing test suite in `src/test.cpp` provides examples of proper test structure. New tests should verify correctness across edge cases including boundary conditions, invalid inputs, and stress testing with large iteration counts.

### Performance Considerations

Performance is a primary design goal for this system. Contributions that impact the game engine core should include benchmark results demonstrating that performance characteristics are maintained or improved. The headless mode provides facilities for automated performance testing.

### Documentation Standards

Technical documentation should be precise and complete. Avoid marketing language, excessive use of formatting, or casual tone. Documentation should explain both what the code does and why particular design decisions were made. Include algorithmic complexity analysis where relevant.

### Plugin Development

Contributions of new example AI implementations should follow the progressive complexity model established by the existing examples. Each new agent should demonstrate a specific technique or algorithm while remaining accessible to students learning AI concepts. Include performance benchmarks and expected win rates against the baseline implementations.

### Submission Process

1. Fork the repository and create a feature branch with a descriptive name
2. Implement changes with appropriate tests and documentation
3. Verify that all existing tests pass and code compiles without warnings
4. Submit a pull request with a detailed description of changes and rationale
5. Address review feedback promptly and professionally

### Licensing

All contributions are subject to the same educational/research license as the main project. By submitting code, contributors affirm that they have the right to contribute the code under these terms and that the contribution does not violate any third-party rights.

## Bug Reports

Bug reports should include:

- Exact steps to reproduce the issue
- Expected behavior versus observed behavior  
- System configuration (OS, compiler version, library versions)
- Relevant log output or error messages
- Minimal test case demonstrating the problem

## Feature Requests

Feature requests should include:

- Clear description of the proposed functionality
- Use case or motivation for the feature
- Consideration of implementation complexity and performance impact
- Compatibility with existing systems and workflows

## Questions and Discussion

For questions about using the system or developing plugins, please refer to the existing documentation (README.md, HEADLESS_MODE.md, PLUGIN_SYSTEM.md, STUDENT_GUIDE.md) before opening an issue. The documentation is comprehensive and addresses most common questions.

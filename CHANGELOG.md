# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2026-02-19

### Added
- Complete C++ rewrite of entire codebase for performance
- Headless execution mode with multi-threading support
- Plugin system for external AI implementations
- Comprehensive CLI with tournament, benchmark, and evaluation modes
- CSV and JSON export for game results
- Five example AI implementations (Random, Material, Tactical, Positional, Aggressive)
- Student template and comprehensive educational documentation
- Automated tournament system with round-robin support
- Performance optimizations achieving 20,000+ games/second for simple agents
- Unit test suite for move generation and game logic
- Cross-platform build system using CMake

### Changed
- Complete rewrite from Python to C++17
- Migrated from Pygame to SDL2 for rendering
- Redesigned AI interface with abstract base classes
- Improved move generation algorithm with validation
- Enhanced evaluation function with positional factors

### Removed
- Python implementation (archived in python_backup directory)
- Legacy bot implementations
- Campaign mode (may be reintroduced in future version)

### Performance
- Random vs Random: 20,000+ games/second (200x faster than Python)
- Greedy vs Greedy: 13,670 games/second
- Plugin agents: 1,000+ games/second
- Minimax depth-2: ~10 games/second
- Alpha-beta depth-3: 2-5 games/second

## [1.0.0] - 2024-2025

### Added
- Initial Python implementation with Pygame
- Basic AI opponents (Random, Apprentice, Novice, Veteran, Champion)
- Graphical user interface with mouse controls
- Tournament mode
- Campaign mode with AI progression
- Basic simulation capabilities

### Implementation
- Python 3.x with Pygame for rendering
- Object-oriented design with separate modules
- Manual move generation and validation
- Simple material-based evaluation

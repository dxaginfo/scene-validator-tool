# SceneValidator

## Overview
SceneValidator is a media validation tool that checks and validates scene metadata and structure in media files. It ensures that scenes conform to required specifications and industry standards.

## Features
- Validates scene metadata against requirements
- Checks scene structure against specifications
- Uses Gemini API for intelligent validation of complex scene elements
- Generates detailed validation reports in multiple formats (JSON, XML, TXT)
- Supports custom validation rules

## Installation

```bash
# Clone the repository
git clone https://github.com/dxaginfo/scene-validator-tool.git
cd scene-validator-tool

# Install dependencies
pip install -r requirements.txt

# Set up Gemini API credentials
export GEMINI_API_KEY=your-api-key
```

## Usage

### Command Line
```bash
python scene_validator.py --file path/to/scene.json --rules path/to/rules.yaml --output json --strict
```

### Python API
```python
from scene_validator import SceneValidator

# Define validation rules
validation_rules = {
    "metadata_requirements": ["title", "description", "creation_date"],
    "structure_requirements": ["scene_boundaries", "proper_nesting"],
    "custom_rules": ["narrative_consistency"]
}

# Initialize validator
validator = SceneValidator()

# Validate a scene
results = validator.validate_scene(
    "path/to/scene.json", 
    validation_rules, 
    output_format="json", 
    strict_mode=True
)
print(results)
```

## Validation Rules
Create a `rules.yaml` file:
```yaml
validation_rules:
  metadata_requirements:
    - title
    - description
    - creation_date
    - author
  structure_requirements:
    - scene_boundaries
    - proper_nesting
    - timeline_integrity
  custom_rules:
    - narrative_consistency
    - character_continuity
```

## Integration with Other Tools
SceneValidator can be easily integrated with other tools in the collection:
- **TimelineAssembler**: Validate scenes before assembly
- **ContinuityTracker**: Provide validation data for continuity checking
- **FormatNormalizer**: Validate scenes after format normalization

## Error Codes
- `SV001`: Missing required metadata field
- `SV002`: Invalid metadata format
- `SV003`: Missing required structure element
- `SV004`: Invalid structure format
- `SV005`: Narrative inconsistency detected
- `SV006`: Character continuity error
- `SV900`: Gemini API integration error

## Sample Scene File
```json
{
  "title": "Opening Scene",
  "description": "Hero walks through abandoned building",
  "creation_date": "2025-06-20",
  "author": "John Doe",
  "elements": [
    {
      "type": "camera",
      "position": {"x": 0, "y": 1.7, "z": 0},
      "rotation": {"x": 0, "y": 0, "z": 0}
    },
    {
      "type": "character",
      "name": "Hero",
      "position": {"x": 2, "y": 0, "z": 1},
      "animation": "walking"
    },
    {
      "type": "environment",
      "name": "AbandonedBuilding",
      "lighting": "dim",
      "atmosphere": "dusty"
    }
  ],
  "timeline": [
    {"time": 0, "event": "Hero enters from right"},
    {"time": 5, "event": "Hero stops and looks around"},
    {"time": 10, "event": "Camera pans to follow Hero"}
  ]
}
```

## License
MIT

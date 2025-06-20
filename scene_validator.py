#!/usr/bin/env python3
"""
SceneValidator - Media Validation Tool

A tool that validates scene metadata and structure in media files,
ensuring they conform to required specifications and industry standards.

Implemented using Python + Gemini API
"""

import argparse
import json
import logging
import os
import sys
import yaml

# When available, use the Gemini API client
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logging.warning("Google Gemini API client not available. Advanced validation disabled.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("SceneValidator")

class SceneValidator:
    """Main class for validating scene files"""
    
    def __init__(self, api_key=None):
        """Initialize the validator
        
        Args:
            api_key: Gemini API key (optional, can be set via env var GEMINI_API_KEY)
        """
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.gemini_client = None
        
        # Initialize Gemini client if available
        if GEMINI_AVAILABLE and self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                self.gemini_client = genai
                logger.info("Gemini API client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini API client: {str(e)}")
    
    def parse_scene_file(self, scene_file):
        """Parse the scene file into structured data
        
        Args:
            scene_file: Path to scene file or URL
            
        Returns:
            dict: Parsed scene data
        """
        try:
            with open(scene_file, 'r') as f:
                if scene_file.endswith('.json'):
                    return json.load(f)
                elif scene_file.endswith('.yaml') or scene_file.endswith('.yml'):
                    return yaml.safe_load(f)
                else:
                    # Try to guess format
                    try:
                        return json.load(f)
                    except json.JSONDecodeError:
                        f.seek(0)  # Reset file pointer
                        return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error parsing scene file: {str(e)}")
            raise
    
    def validate_metadata(self, scene_data, metadata_requirements):
        """Validate scene metadata against requirements
        
        Args:
            scene_data: Parsed scene data
            metadata_requirements: List of required metadata fields
            
        Returns:
            dict: Validation results
        """
        validation_results = {
            "errors": [],
            "warnings": [],
            "validation_summary": {
                "total_checks": len(metadata_requirements),
                "passed_checks": 0,
                "failed_checks": 0,
                "warning_count": 0
            }
        }
        
        for field in metadata_requirements:
            if field not in scene_data:
                validation_results["errors"].append({
                    "error_code": "SV001",
                    "error_description": f"Missing required metadata field: {field}",
                    "location": "metadata",
                    "severity": "critical"
                })
                validation_results["validation_summary"]["failed_checks"] += 1
            else:
                validation_results["validation_summary"]["passed_checks"] += 1
        
        return validation_results
    
    def validate_structure(self, scene_data, structure_requirements):
        """Validate scene structure against specifications
        
        Args:
            scene_data: Parsed scene data
            structure_requirements: List of required structural elements
            
        Returns:
            dict: Validation results
        """
        validation_results = {
            "errors": [],
            "warnings": [],
            "validation_summary": {
                "total_checks": len(structure_requirements),
                "passed_checks": 0,
                "failed_checks": 0,
                "warning_count": 0
            }
        }
        
        for requirement in structure_requirements:
            if requirement == "scene_boundaries":
                if "elements" not in scene_data:
                    validation_results["errors"].append({
                        "error_code": "SV003",
                        "error_description": "Missing required structure element: elements",
                        "location": "structure",
                        "severity": "critical"
                    })
                    validation_results["validation_summary"]["failed_checks"] += 1
                else:
                    validation_results["validation_summary"]["passed_checks"] += 1
            
            elif requirement == "proper_nesting":
                if "elements" in scene_data:
                    for i, element in enumerate(scene_data["elements"]):
                        if "type" not in element:
                            validation_results["errors"].append({
                                "error_code": "SV004",
                                "error_description": f"Element at index {i} missing type attribute",
                                "location": f"elements[{i}]",
                                "severity": "critical"
                            })
                            validation_results["validation_summary"]["failed_checks"] += 1
                            continue
                    validation_results["validation_summary"]["passed_checks"] += 1
                else:
                    validation_results["validation_summary"]["failed_checks"] += 1
            
            elif requirement == "timeline_integrity":
                if "timeline" in scene_data:
                    # Check if timeline events are in chronological order
                    times = [event.get("time") for event in scene_data["timeline"] if "time" in event]
                    if times and times != sorted(times):
                        validation_results["errors"].append({
                            "error_code": "SV004",
                            "error_description": "Timeline events are not in chronological order",
                            "location": "timeline",
                            "severity": "warning"
                        })
                        validation_results["validation_summary"]["warning_count"] += 1
                    else:
                        validation_results["validation_summary"]["passed_checks"] += 1
                else:
                    validation_results["warnings"].append({
                        "warning_code": "SV101",
                        "warning_description": "No timeline found in scene",
                        "location": "structure",
                        "recommendation": "Add a timeline section to improve scene completeness"
                    })
                    validation_results["validation_summary"]["warning_count"] += 1
        
        return validation_results
    
    def create_validation_prompt(self, scene_data, rule):
        """Create a prompt for Gemini API based on the custom rule
        
        Args:
            scene_data: Parsed scene data
            rule: The custom rule to validate
            
        Returns:
            str: Prompt for Gemini
        """
        if rule == "narrative_consistency":
            prompt = f"""
            Analyze this scene for narrative consistency. Identify any narrative inconsistencies.
            Return your analysis in JSON format with these fields:
            - is_consistent (boolean): true if the scene is narratively consistent, false otherwise
            - issues (array): list of any narrative inconsistencies found
            - severity (string): "none", "minor", "major", or "critical"
            
            Scene data:
            {json.dumps(scene_data, indent=2)}
            """
        elif rule == "character_continuity":
            prompt = f"""
            Analyze this scene for character continuity issues. Check if character positions, 
            actions and attributes remain consistent throughout the scene timeline.
            Return your analysis in JSON format with these fields:
            - has_continuity_issues (boolean): true if there are continuity issues, false otherwise
            - issues (array): list of any continuity issues found
            - severity (string): "none", "minor", "major", or "critical"
            
            Scene data:
            {json.dumps(scene_data, indent=2)}
            """
        else:
            prompt = f"""
            Analyze this scene data for issues related to the rule: {rule}.
            Return your analysis in JSON format with these fields:
            - has_issues (boolean): true if there are issues, false otherwise
            - issues (array): list of any issues found
            - severity (string): "none", "minor", "major", or "critical"
            
            Scene data:
            {json.dumps(scene_data, indent=2)}
            """
        
        return prompt
    
    def parse_gemini_response(self, response, rule):
        """Parse Gemini API response into validation results format
        
        Args:
            response: Response from Gemini API
            rule: The custom rule that was validated
            
        Returns:
            dict: Validation results
        """
        validation_results = {
            "errors": [],
            "warnings": [],
            "validation_summary": {
                "total_checks": 1,
                "passed_checks": 0,
                "failed_checks": 0,
                "warning_count": 0
            }
        }
        
        try:
            # Convert Gemini response to JSON
            result = json.loads(response.text)
            
            # Process based on rule type
            if rule == "narrative_consistency":
                if not result.get("is_consistent", True):
                    for issue in result.get("issues", []):
                        validation_results["errors" if result.get("severity") == "critical" else "warnings"].append({
                            "error_code" if result.get("severity") == "critical" else "warning_code": "SV005",
                            "error_description" if result.get("severity") == "critical" else "warning_description": issue,
                            "location": "narrative",
                            "severity" if result.get("severity") == "critical" else "recommendation": result.get("severity", "info")
                        })
                    
                    if result.get("severity") == "critical":
                        validation_results["validation_summary"]["failed_checks"] += 1
                    else:
                        validation_results["validation_summary"]["warning_count"] += 1
                else:
                    validation_results["validation_summary"]["passed_checks"] += 1
            
            elif rule == "character_continuity":
                if result.get("has_continuity_issues", False):
                    for issue in result.get("issues", []):
                        validation_results["errors" if result.get("severity") == "critical" else "warnings"].append({
                            "error_code" if result.get("severity") == "critical" else "warning_code": "SV006",
                            "error_description" if result.get("severity") == "critical" else "warning_description": issue,
                            "location": "character_continuity",
                            "severity" if result.get("severity") == "critical" else "recommendation": result.get("severity", "info")
                        })
                    
                    if result.get("severity") == "critical":
                        validation_results["validation_summary"]["failed_checks"] += 1
                    else:
                        validation_results["validation_summary"]["warning_count"] += 1
                else:
                    validation_results["validation_summary"]["passed_checks"] += 1
            
            else:  # Generic rule handling
                if result.get("has_issues", False):
                    for issue in result.get("issues", []):
                        validation_results["errors" if result.get("severity") == "critical" else "warnings"].append({
                            "error_code" if result.get("severity") == "critical" else "warning_code": "SV007",
                            "error_description" if result.get("severity") == "critical" else "warning_description": issue,
                            "location": rule,
                            "severity" if result.get("severity") == "critical" else "recommendation": result.get("severity", "info")
                        })
                    
                    if result.get("severity") == "critical":
                        validation_results["validation_summary"]["failed_checks"] += 1
                    else:
                        validation_results["validation_summary"]["warning_count"] += 1
                else:
                    validation_results["validation_summary"]["passed_checks"] += 1
                    
        except Exception as e:
            logger.error(f"Error parsing Gemini response: {str(e)}")
            validation_results["warnings"].append({
                "warning_code": "SV900",
                "warning_description": f"Failed to parse Gemini response for rule '{rule}'",
                "location": "gemini_integration",
                "recommendation": "Check Gemini API connectivity and response format"
            })
            validation_results["validation_summary"]["warning_count"] += 1
        
        return validation_results
    
    def validate_with_gemini(self, scene_data, custom_rules):
        """Perform validation using Gemini API
        
        Args:
            scene_data: Parsed scene data
            custom_rules: List of custom validation rules
            
        Returns:
            dict: Validation results
        """
        validation_results = {
            "errors": [],
            "warnings": [],
            "validation_summary": {
                "total_checks": len(custom_rules),
                "passed_checks": 0,
                "failed_checks": 0,
                "warning_count": 0
            }
        }
        
        # Skip if Gemini is not available
        if not GEMINI_AVAILABLE or not self.gemini_client:
            validation_results["warnings"].append({
                "warning_code": "SV900",
                "warning_description": "Gemini API not available for advanced validation",
                "location": "gemini_integration",
                "recommendation": "Install Gemini API client and set up API key"
            })
            validation_results["validation_summary"]["warning_count"] += len(custom_rules)
            return validation_results
        
        # For each custom rule, use Gemini to validate
        for rule in custom_rules:
            # Create prompt for Gemini
            prompt = self.create_validation_prompt(scene_data, rule)
            
            try:
                # Get response from Gemini API
                model = self.gemini_client.GenerativeModel('gemini-pro')
                response = model.generate_content(prompt)
                
                # Parse Gemini response
                rule_result = self.parse_gemini_response(response, rule)
                
                # Update validation results
                self.update_validation_results(validation_results, rule_result)
            except Exception as e:
                logger.error(f"Error validating with Gemini: {str(e)}")
                validation_results["warnings"].append({
                    "warning_code": "SV901",
                    "warning_description": f"Failed to validate rule '{rule}' with Gemini",
                    "location": "gemini_integration",
                    "recommendation": f"Check Gemini API connectivity: {str(e)}"
                })
                validation_results["validation_summary"]["warning_count"] += 1
        
        return validation_results
    
    def update_validation_results(self, base_results, new_results):
        """Update base validation results with new results
        
        Args:
            base_results: Base validation results to update
            new_results: New results to incorporate
        """
        # Add errors and warnings
        base_results["errors"].extend(new_results.get("errors", []))
        base_results["warnings"].extend(new_results.get("warnings", []))
        
        # Update summary counts
        summary = base_results["validation_summary"]
        new_summary = new_results.get("validation_summary", {})
        
        summary["total_checks"] += new_summary.get("total_checks", 0)
        summary["passed_checks"] += new_summary.get("passed_checks", 0)
        summary["failed_checks"] += new_summary.get("failed_checks", 0)
        summary["warning_count"] += new_summary.get("warning_count", 0)
    
    def format_results(self, validation_results, output_format="json"):
        """Format validation results based on requested format
        
        Args:
            validation_results: Validation results to format
            output_format: Desired output format (json, xml, txt)
            
        Returns:
            str: Formatted results
        """
        if output_format.lower() == "json":
            return json.dumps(validation_results, indent=2)
        elif output_format.lower() == "xml":
            # Simple XML formatting (would use a proper library in production)
            xml = ['<?xml version="1.0" encoding="UTF-8"?>\n<validation>']
            
            xml.append(f'<validation_status>{validation_results["validation_status"]}</validation_status>')
            
            # Add errors
            xml.append('<errors>')
            for error in validation_results.get("errors", []):
                xml.append('  <error>')
                for k, v in error.items():
                    xml.append(f'    <{k}>{v}</{k}>')
                xml.append('  </error>')
            xml.append('</errors>')
            
            # Add warnings
            xml.append('<warnings>')
            for warning in validation_results.get("warnings", []):
                xml.append('  <warning>')
                for k, v in warning.items():
                    xml.append(f'    <{k}>{v}</{k}>')
                xml.append('  </warning>')
            xml.append('</warnings>')
            
            # Add summary
            xml.append('<validation_summary>')
            for k, v in validation_results.get("validation_summary", {}).items():
                xml.append(f'  <{k}>{v}</{k}>')
            xml.append('</validation_summary>')
            
            xml.append('</validation>')
            return '\n'.join(xml)
        else:  # text format
            lines = []
            lines.append("=== Scene Validation Results ===")
            lines.append(f"Validation Status: {validation_results.get('validation_status', 'unknown')}")
            
            if validation_results.get("errors", []):
                lines.append("\nErrors:")
                for i, error in enumerate(validation_results["errors"], 1):
                    lines.append(f"  {i}. {error.get('error_description')}")
                    lines.append(f"     Code: {error.get('error_code')}")
                    lines.append(f"     Location: {error.get('location')}")
                    lines.append(f"     Severity: {error.get('severity')}")
            
            if validation_results.get("warnings", []):
                lines.append("\nWarnings:")
                for i, warning in enumerate(validation_results["warnings"], 1):
                    lines.append(f"  {i}. {warning.get('warning_description')}")
                    lines.append(f"     Code: {warning.get('warning_code')}")
                    lines.append(f"     Location: {warning.get('location')}")
                    lines.append(f"     Recommendation: {warning.get('recommendation')}")
            
            if "validation_summary" in validation_results:
                summary = validation_results["validation_summary"]
                lines.append("\nSummary:")
                lines.append(f"  Total Checks: {summary.get('total_checks', 0)}")
                lines.append(f"  Passed Checks: {summary.get('passed_checks', 0)}")
                lines.append(f"  Failed Checks: {summary.get('failed_checks', 0)}")
                lines.append(f"  Warning Count: {summary.get('warning_count', 0)}")
                
                if summary.get('total_checks', 0) > 0:
                    pass_rate = (summary.get('passed_checks', 0) / summary.get('total_checks', 0)) * 100
                    lines.append(f"  Pass Rate: {pass_rate:.1f}%")
            
            return '\n'.join(lines)
    
    def validate_scene(self, scene_file, validation_rules, output_format="json", strict_mode=False):
        """Main validation function
        
        Args:
            scene_file: Path to scene file or URL
            validation_rules: Dictionary of validation rules
            output_format: Desired output format
            strict_mode: Whether to use strict validation mode
            
        Returns:
            str: Formatted validation results
        """
        # Parse the scene file
        scene_data = self.parse_scene_file(scene_file)
        
        # Initialize validation results
        validation_results = {
            "errors": [],
            "warnings": [],
            "validation_summary": {
                "total_checks": 0,
                "passed_checks": 0,
                "failed_checks": 0,
                "warning_count": 0
            }
        }
        
        # Validate metadata
        if "metadata_requirements" in validation_rules:
            metadata_results = self.validate_metadata(scene_data, validation_rules["metadata_requirements"])
            self.update_validation_results(validation_results, metadata_results)
        
        # Validate structure
        if "structure_requirements" in validation_rules:
            structure_results = self.validate_structure(scene_data, validation_rules["structure_requirements"])
            self.update_validation_results(validation_results, structure_results)
        
        # Use Gemini API for complex validations
        if "custom_rules" in validation_rules and validation_rules["custom_rules"]:
            gemini_results = self.validate_with_gemini(scene_data, validation_rules["custom_rules"])
            self.update_validation_results(validation_results, gemini_results)
        
        # Determine overall validation status
        if validation_results["errors"] and strict_mode:
            validation_results["validation_status"] = "invalid"
        elif validation_results["errors"]:
            validation_results["validation_status"] = "warning"
        else:
            validation_results["validation_status"] = "valid"
        
        # Format and return results
        return self.format_results(validation_results, output_format)


def load_rules_from_file(rules_file):
    """Load validation rules from a file
    
    Args:
        rules_file: Path to rules file (YAML or JSON)
        
    Returns:
        dict: Validation rules
    """
    try:
        with open(rules_file, 'r') as f:
            if rules_file.endswith('.json'):
                return json.load(f)
            elif rules_file.endswith('.yaml') or rules_file.endswith('.yml'):
                return yaml.safe_load(f)
            else:
                # Try to guess format
                try:
                    return json.load(f)
                except json.JSONDecodeError:
                    f.seek(0)  # Reset file pointer
                    return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Error loading rules file: {str(e)}")
        raise


def main():
    """Main entry point for command line usage"""
    parser = argparse.ArgumentParser(description="Validate scene metadata and structure in media files")
    parser.add_argument("--file", required=True, help="Path to scene file to validate")
    parser.add_argument("--rules", required=True, help="Path to validation rules file")
    parser.add_argument("--output", choices=["json", "xml", "txt"], default="json", 
                        help="Output format (default: json)")
    parser.add_argument("--strict", action="store_true", help="Enable strict validation mode")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    # Load validation rules
    validation_rules = load_rules_from_file(args.rules)
    
    # Initialize validator
    validator = SceneValidator()
    
    # Validate scene
    results = validator.validate_scene(
        args.file,
        validation_rules,
        args.output,
        args.strict
    )
    
    print(results)
    
    # Return exit code based on validation status
    if "invalid" in results:
        return 1
    else:
        return 0


if __name__ == "__main__":
    sys.exit(main())

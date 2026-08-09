import re

class LogAnalyzer:
    """
    Utility class to parse massive CI/CD logs and extract just the relevant error traces 
    before passing them to the expensive LLM.
    """
    
    @staticmethod
    def extract_error_trace(raw_log: str) -> str:
        """
        Extremely naive but effective log parser. Looks for common error signatures
        across Python, Node, and compiler logs to clip out the noise.
        """
        lines = raw_log.split('\n')
        error_lines = []
        capture_mode = False
        
        # Common signals that an error trace has started
        error_keywords = [
            "Error:", "Exception:", "Traceback (most recent call last):", 
            "npm ERR!", "FAIL", "FAILED", "AssertionError"
        ]
        
        for line in lines:
            # Drop obvious github action timestamps to save context
            clean_line = re.sub(r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z\s', '', line)
            
            if not capture_mode:
                for keyword in error_keywords:
                    if keyword in clean_line:
                        capture_mode = True
                        error_lines.append("--- BEGIN ERROR TRACE ---")
                        error_lines.append(clean_line)
                        break
            else:
                error_lines.append(clean_line)
                
                # Stop capturing after 50 lines to prevent context bloat, 
                # or if we hit a success marker
                if len(error_lines) > 50 or "Done in" in clean_line or "info Visit" in clean_line:
                    break
                    
        return '\n'.join(error_lines) if error_lines else raw_log[-2000:] # Fallback to last 2000 chars

log_analyzer = LogAnalyzer()

"""Confidence calculation utilities for agent outputs.

Agents report confidence based on observable signals about data quality,
not knowledge of perturbations. This enables organic control flow changes
when data quality varies.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import re
import json


@dataclass
class ConfidenceFactors:
    """Individual factors contributing to overall confidence."""
    parsing_quality: float = 1.0      # Could we parse/read the data cleanly?
    data_completeness: float = 1.0    # Is the data complete or are things missing?
    result_consistency: float = 1.0   # Are results internally consistent?
    answer_specificity: float = 1.0   # Is the answer concrete or vague?
    execution_success: float = 1.0    # Did execution succeed without errors?

    def to_dict(self) -> Dict[str, float]:
        return {
            "parsing_quality": round(self.parsing_quality, 2),
            "data_completeness": round(self.data_completeness, 2),
            "result_consistency": round(self.result_consistency, 2),
            "answer_specificity": round(self.answer_specificity, 2),
            "execution_success": round(self.execution_success, 2),
        }

    def overall(self, weights: Optional[Dict[str, float]] = None) -> float:
        """Calculate weighted overall confidence score."""
        if weights is None:
            # Default equal weights
            weights = {
                "parsing_quality": 0.25,
                "data_completeness": 0.20,
                "result_consistency": 0.20,
                "answer_specificity": 0.20,
                "execution_success": 0.15,
            }

        score = (
            self.parsing_quality * weights.get("parsing_quality", 0.2) +
            self.data_completeness * weights.get("data_completeness", 0.2) +
            self.result_consistency * weights.get("result_consistency", 0.2) +
            self.answer_specificity * weights.get("answer_specificity", 0.2) +
            self.execution_success * weights.get("execution_success", 0.2)
        )
        return round(min(1.0, max(0.0, score)), 2)


@dataclass
class ConfidenceResult:
    """Result of confidence calculation."""
    score: float
    factors: ConfidenceFactors
    signals: List[str] = field(default_factory=list)  # Human-readable signals

    def to_dict(self) -> Dict[str, Any]:
        return {
            "confidence": self.score,
            "confidence_factors": self.factors.to_dict(),
            "confidence_signals": self.signals,
        }


def calculate_text_parsing_confidence(text: str) -> tuple[float, List[str]]:
    """Assess parsing quality based on text characteristics.

    Returns (confidence, signals) where signals explain the assessment.
    """
    if not text or not text.strip():
        return 0.0, ["empty_content"]

    signals = []
    score = 1.0

    # Check for encoding artifacts (generic checks only)
    # These detect clearly broken text, not subtle corruptions (LLM handles those)
    garbled_patterns = [
        r'[^\x00-\x7F]{3,}',  # Multiple non-ASCII chars in a row (encoding issues)
        r'\?{3,}',            # Multiple question marks (failed decoding)
        r'[�]{1,}',           # Replacement characters
        r'[\x00-\x08\x0b\x0c\x0e-\x1f]{2,}',  # Control characters
    ]

    garbled_count = 0
    for pattern in garbled_patterns:
        matches = re.findall(pattern, text)
        garbled_count += len(matches)

    if garbled_count > 10:
        score -= 0.4
        signals.append(f"many_encoding_errors:{garbled_count}")
    elif garbled_count > 3:
        score -= 0.2
        signals.append(f"some_encoding_errors:{garbled_count}")
    elif garbled_count > 0:
        score -= 0.1
        signals.append(f"few_encoding_errors:{garbled_count}")

    # Check for excessive special characters (noise)
    special_ratio = len(re.findall(r'[^\w\s.,;:!?\'"()-]', text)) / max(len(text), 1)
    if special_ratio > 0.1:
        score -= 0.2
        signals.append(f"high_special_char_ratio:{special_ratio:.2f}")

    # Check for very short content (might be truncated)
    if len(text.strip()) < 50:
        score -= 0.2
        signals.append("very_short_content")

    # Check for repetitive patterns (might indicate parsing issues)
    words = text.split()
    if len(words) > 10:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.3:
            score -= 0.2
            signals.append(f"highly_repetitive:{unique_ratio:.2f}")

    return max(0.0, score), signals


def calculate_data_completeness_confidence(
    expected_fields: Optional[int] = None,
    actual_fields: Optional[int] = None,
    null_ratio: Optional[float] = None,
    error_messages: Optional[List[str]] = None,
) -> tuple[float, List[str]]:
    """Assess data completeness.

    Returns (confidence, signals).
    """
    signals = []
    score = 1.0

    # Check field completeness
    if expected_fields and actual_fields is not None:
        field_ratio = actual_fields / expected_fields
        if field_ratio < 0.5:
            score -= 0.4
            signals.append(f"missing_many_fields:{actual_fields}/{expected_fields}")
        elif field_ratio < 0.8:
            score -= 0.2
            signals.append(f"missing_some_fields:{actual_fields}/{expected_fields}")

    # Check null ratio
    if null_ratio is not None:
        if null_ratio > 0.5:
            score -= 0.3
            signals.append(f"high_null_ratio:{null_ratio:.2f}")
        elif null_ratio > 0.2:
            score -= 0.15
            signals.append(f"moderate_null_ratio:{null_ratio:.2f}")

    # Check for error indicators in messages
    if error_messages:
        error_keywords = ["not found", "missing", "unavailable", "empty", "null", "none"]
        for msg in error_messages:
            msg_lower = msg.lower()
            if any(kw in msg_lower for kw in error_keywords):
                score -= 0.1
                signals.append("error_indicates_missing_data")
                break

    return max(0.0, score), signals


def calculate_answer_specificity_confidence(answer: str) -> tuple[float, List[str]]:
    """Assess how specific/concrete an answer is.

    Returns (confidence, signals).
    """
    if not answer or not answer.strip():
        return 0.0, ["no_answer"]

    signals = []
    score = 1.0
    answer_lower = answer.lower()

    # Check for hedging language
    hedging_phrases = [
        "i think", "probably", "maybe", "possibly", "might be",
        "could be", "seems like", "appears to", "i believe",
        "not sure", "uncertain", "unclear", "approximately",
        "around", "roughly", "estimate", "guess"
    ]

    hedging_count = sum(1 for phrase in hedging_phrases if phrase in answer_lower)
    if hedging_count >= 3:
        score -= 0.4
        signals.append(f"much_hedging:{hedging_count}")
    elif hedging_count >= 1:
        score -= 0.15 * hedging_count
        signals.append(f"some_hedging:{hedging_count}")

    # Check for inability indicators
    inability_phrases = [
        "cannot determine", "unable to", "not possible",
        "insufficient", "no way to", "cannot find"
    ]
    if any(phrase in answer_lower for phrase in inability_phrases):
        score -= 0.4
        signals.append("indicates_inability")

    # Check for concrete values (numbers, specific terms)
    has_number = bool(re.search(r'\b\d+\.?\d*\b', answer))
    has_specific_terms = len(answer.split()) > 3  # More than just "I don't know"

    if has_number:
        score += 0.1  # Bonus for concrete numeric answer
        signals.append("has_numeric_value")

    if not has_specific_terms:
        score -= 0.2
        signals.append("very_vague_answer")

    return max(0.0, min(1.0, score)), signals


def calculate_execution_confidence(
    success: bool,
    attempts: int = 1,
    error_message: Optional[str] = None,
    warnings: Optional[List[str]] = None,
) -> tuple[float, List[str]]:
    """Assess execution success.

    Returns (confidence, signals).
    """
    signals = []

    if not success:
        signals.append("execution_failed")
        if error_message:
            signals.append(f"error:{error_message[:50]}")
        return 0.0, signals

    score = 1.0

    # Penalize multiple attempts
    if attempts > 3:
        score -= 0.3
        signals.append(f"many_retries:{attempts}")
    elif attempts > 1:
        score -= 0.1 * (attempts - 1)
        signals.append(f"some_retries:{attempts}")

    # Check for warnings
    if warnings:
        warning_keywords = ["warning", "deprecated", "error", "failed"]
        warning_count = sum(
            1 for w in warnings
            if any(kw in w.lower() for kw in warning_keywords)
        )
        if warning_count > 0:
            score -= 0.1 * min(warning_count, 3)
            signals.append(f"has_warnings:{warning_count}")

    return max(0.0, score), signals


def assess_data_quality_with_llm(
    text: str,
    llm,
    data_type: str = "document",
    max_chars: int = 3000,
) -> tuple[float, List[str]]:
    """Use LLM to assess the quality of extracted/parsed data.

    This provides semantic quality assessment that can detect issues
    rule-based checks miss, such as:
    - Redacted or placeholder content
    - Corrupted but syntactically valid text
    - Missing or incomplete information
    - Data that "looks wrong" semantically

    Args:
        text: The extracted text to assess
        llm: LLM client for generating assessment
        data_type: Type of data (document, table, transcript, etc.)
        max_chars: Maximum characters to send to LLM

    Returns:
        (confidence_score, list_of_issues)
    """
    if not text or not text.strip():
        return 0.0, ["empty_content"]

    # Truncate if needed
    sample = text[:max_chars]
    if len(text) > max_chars:
        sample += "\n[... truncated ...]"

    prompt = f"""Assess the quality of this extracted {data_type} text.

Look for ANY signs of data quality issues such as:
- Garbled, corrupted, or unreadable text
- Placeholder markers (e.g., [NAME], [REDACTED], ???, N/A in unexpected places)
- Characters that look wrong or out of place
- Missing or incomplete information
- Text that doesn't make semantic sense

TEXT TO ASSESS:
---
{sample}
---

Respond with a JSON object:
{{"score": <0.0 to 1.0>, "issues": ["issue1", "issue2", ...]}}

Where score is:
- 0.9-1.0 = Clean, high-quality data with no issues
- 0.6-0.8 = 1-2 minor issues that don't affect understanding
- 0.4-0.5 = Multiple issues OR any corruption/garbling that could cause misinterpretation
- 0.0-0.3 = Severe issues, data is unreliable or unusable

IMPORTANT: Be strict. If you find character substitutions (like '1' for 'l', '0' for 'o', '5' for 's'),
garbled words, or multiple formatting issues, score should be 0.5 or lower.
Any corruption that could lead to wrong answers should score below 0.5.

Only list actual issues found. If the data looks clean, return {{"score": 1.0, "issues": []}}

JSON response:"""

    try:
        response = llm.generate(prompt, response_type="quality_check")
        content = response.content.strip()

        # Try to parse JSON from response
        # Handle cases where LLM wraps in markdown code blocks
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        result = json.loads(content)
        score = float(result.get("score", 0.5))
        issues = result.get("issues", [])

        # Ensure score is in valid range
        score = max(0.0, min(1.0, score))

        return score, issues

    except (json.JSONDecodeError, KeyError, ValueError) as e:
        # If LLM response parsing fails, return moderate confidence
        return 0.7, [f"quality_check_parse_error:{str(e)[:50]}"]
    except Exception as e:
        # For other errors, don't fail the whole process
        return 0.7, [f"quality_check_error:{str(e)[:50]}"]


def calculate_tabular_data_confidence(
    shape: tuple,
    null_count: int = 0,
    execution_success: bool = True,
    attempts: int = 1,
    output_text: str = "",
    raw_text: str = "",
    llm=None,
) -> ConfidenceResult:
    """Calculate confidence for tabular data analysis.

    Uses LLM-based semantic assessment as the PRIMARY signal (50% weight),
    combined with other factors for a complete assessment.

    Args:
        shape: (rows, cols) tuple
        null_count: Number of null/empty cells
        execution_success: Whether execution succeeded
        attempts: Number of attempts made
        output_text: The agent's output/answer text
        raw_text: Raw extracted text for LLM quality check
        llm: LLM client for semantic quality assessment
    """
    factors = ConfidenceFactors()
    all_signals = []

    # LLM assessment is the primary signal (will be weighted at 50%)
    if llm and raw_text:
        llm_score, llm_issues = assess_data_quality_with_llm(
            raw_text, llm, data_type="tabular data"
        )
        factors.parsing_quality = llm_score
        if llm_issues:
            all_signals.extend([f"llm:{issue}" for issue in llm_issues])
    else:
        factors.parsing_quality = 1.0

    # Data completeness
    total_cells = shape[0] * shape[1] if shape[0] > 0 and shape[1] > 0 else 1
    null_ratio = null_count / total_cells
    comp_score, comp_signals = calculate_data_completeness_confidence(null_ratio=null_ratio)
    factors.data_completeness = comp_score
    all_signals.extend(comp_signals)

    # Execution success
    exec_score, exec_signals = calculate_execution_confidence(
        success=execution_success,
        attempts=attempts,
    )
    factors.execution_success = exec_score
    all_signals.extend(exec_signals)

    # Answer specificity from output
    if output_text:
        spec_score, spec_signals = calculate_answer_specificity_confidence(output_text)
        factors.answer_specificity = spec_score
        all_signals.extend(spec_signals)

    # Custom weights: LLM (parsing_quality) gets 50%, rest split among others
    weights = {
        "parsing_quality": 0.60,      # LLM assessment - primary signal
        "data_completeness": 0.05,
        "result_consistency": 0.05,   # Not really used for tabular
        "answer_specificity": 0.15,
        "execution_success": 0.15,
    }

    return ConfidenceResult(
        score=factors.overall(weights),
        factors=factors,
        signals=all_signals,
    )


def calculate_document_confidence(
    text: str,
    page_count: int = 1,
    extraction_success: bool = True,
    output_text: str = "",
    llm=None,
) -> ConfidenceResult:
    """Calculate confidence for document analysis (PDF, DOCX, etc.).

    Uses LLM-based semantic assessment as the PRIMARY signal (50% weight),
    combined with other factors for a complete assessment.

    Args:
        text: Extracted document text
        page_count: Number of pages in document
        extraction_success: Whether extraction succeeded
        output_text: The agent's output/answer text
        llm: LLM client for semantic quality assessment
    """
    factors = ConfidenceFactors()
    all_signals = []

    # LLM assessment is the primary signal (will be weighted at 50%)
    if llm and text:
        llm_score, llm_issues = assess_data_quality_with_llm(
            text, llm, data_type="document"
        )
        factors.parsing_quality = llm_score
        if llm_issues:
            all_signals.extend([f"llm:{issue}" for issue in llm_issues])
    else:
        factors.parsing_quality = 1.0

    # Data completeness - check if we got reasonable content
    expected_chars_per_page = 500  # Rough heuristic
    expected_chars = page_count * expected_chars_per_page
    if len(text) < expected_chars * 0.3:
        factors.data_completeness = 0.5
        all_signals.append("less_content_than_expected")
    elif len(text) < expected_chars * 0.6:
        factors.data_completeness = 0.7
        all_signals.append("somewhat_less_content")

    # Execution
    factors.execution_success = 1.0 if extraction_success else 0.0
    if not extraction_success:
        all_signals.append("extraction_failed")

    # Answer specificity
    if output_text:
        spec_score, spec_signals = calculate_answer_specificity_confidence(output_text)
        factors.answer_specificity = spec_score
        all_signals.extend(spec_signals)

    # Custom weights: LLM (parsing_quality) gets 50%, rest split among others
    weights = {
        "parsing_quality": 0.50,      # LLM assessment - primary signal
        "data_completeness": 0.15,
        "result_consistency": 0.05,   # Not really used for documents
        "answer_specificity": 0.15,
        "execution_success": 0.15,
    }

    return ConfidenceResult(
        score=factors.overall(weights),
        factors=factors,
        signals=all_signals,
    )


def calculate_visual_confidence(
    image_loaded: bool,
    response_text: str,
    response_length: int = 0,
    llm=None,
) -> ConfidenceResult:
    """Calculate confidence for visual/image analysis.

    Args:
        image_loaded: Whether the image was loaded successfully
        response_text: The vision model's response/description
        response_length: Length of response
        llm: Optional LLM client for semantic quality assessment
    """
    factors = ConfidenceFactors()
    all_signals = []

    if not image_loaded:
        factors.parsing_quality = 0.0
        all_signals.append("image_load_failed")
        return ConfidenceResult(score=0.0, factors=factors, signals=all_signals)

    factors.parsing_quality = 1.0

    # Check response quality
    if response_length < 20:
        factors.data_completeness = 0.3
        all_signals.append("very_short_response")
    elif response_length < 50:
        factors.data_completeness = 0.6
        all_signals.append("short_response")

    # Check for vision model uncertainty
    uncertainty_phrases = [
        "cannot see", "unclear", "blurry", "hard to read",
        "not visible", "cannot determine", "unable to",
        "illegible", "obscured", "too small"
    ]
    response_lower = response_text.lower()
    uncertainty_count = sum(1 for p in uncertainty_phrases if p in response_lower)

    if uncertainty_count >= 2:
        factors.result_consistency = 0.4
        all_signals.append(f"vision_uncertainty:{uncertainty_count}")
    elif uncertainty_count >= 1:
        factors.result_consistency = 0.7
        all_signals.append("some_vision_uncertainty")

    # LLM-based quality check on vision response (if LLM provided)
    if llm and response_text:
        llm_score, llm_issues = assess_data_quality_with_llm(
            response_text, llm, data_type="image description"
        )
        # Use LLM assessment for result consistency
        factors.result_consistency = (factors.result_consistency + llm_score) / 2
        if llm_issues:
            all_signals.extend([f"llm:{issue}" for issue in llm_issues])

    # Answer specificity
    spec_score, spec_signals = calculate_answer_specificity_confidence(response_text)
    factors.answer_specificity = spec_score
    all_signals.extend(spec_signals)

    factors.execution_success = 1.0

    return ConfidenceResult(
        score=factors.overall(),
        factors=factors,
        signals=all_signals,
    )


def calculate_audio_confidence(
    transcription: str,
    transcription_success: bool,
    audio_duration: Optional[float] = None,
) -> ConfidenceResult:
    """Calculate confidence for audio transcription/analysis."""
    factors = ConfidenceFactors()
    all_signals = []

    if not transcription_success:
        factors.execution_success = 0.0
        all_signals.append("transcription_failed")
        return ConfidenceResult(score=0.0, factors=factors, signals=all_signals)

    factors.execution_success = 1.0

    # Check transcription quality
    parse_score, parse_signals = calculate_text_parsing_confidence(transcription)
    factors.parsing_quality = parse_score
    all_signals.extend(parse_signals)

    # Check for transcription artifacts
    artifacts = ["[inaudible]", "[unclear]", "[music]", "[noise]", "..."]
    artifact_count = sum(transcription.lower().count(a.lower()) for a in artifacts)

    if artifact_count > 5:
        factors.data_completeness = 0.5
        all_signals.append(f"many_transcription_gaps:{artifact_count}")
    elif artifact_count > 2:
        factors.data_completeness = 0.7
        all_signals.append(f"some_transcription_gaps:{artifact_count}")

    # Check length vs expected (if duration known)
    if audio_duration and audio_duration > 0:
        # Rough heuristic: ~150 words per minute, ~5 chars per word
        expected_chars = audio_duration * 150 * 5 / 60
        if len(transcription) < expected_chars * 0.3:
            factors.data_completeness *= 0.6
            all_signals.append("transcription_shorter_than_expected")

    return ConfidenceResult(
        score=factors.overall(),
        factors=factors,
        signals=all_signals,
    )


def calculate_web_search_confidence(
    results_count: int,
    results_relevant: int = 0,
    fetch_success: bool = True,
    content_length: int = 0,
) -> ConfidenceResult:
    """Calculate confidence for web search/fetch results."""
    factors = ConfidenceFactors()
    all_signals = []

    # Execution success
    if not fetch_success:
        factors.execution_success = 0.0
        all_signals.append("fetch_failed")
        return ConfidenceResult(score=0.1, factors=factors, signals=all_signals)

    factors.execution_success = 1.0

    # Data completeness - did we get results?
    if results_count == 0:
        factors.data_completeness = 0.2
        all_signals.append("no_search_results")
    elif results_count < 3:
        factors.data_completeness = 0.6
        all_signals.append("few_search_results")
    else:
        factors.data_completeness = 1.0

    # Check content quality
    if content_length < 100:
        factors.parsing_quality = 0.5
        all_signals.append("very_short_content")
    elif content_length < 500:
        factors.parsing_quality = 0.7
        all_signals.append("short_content")

    return ConfidenceResult(
        score=factors.overall(),
        factors=factors,
        signals=all_signals,
    )

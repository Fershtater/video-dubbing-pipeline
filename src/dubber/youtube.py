"""
YouTube description and chapters generator.
"""
import logging
from pathlib import Path
from typing import Union

from .models import Chapter, Segment, YouTubeAssets

logger = logging.getLogger("dubber")

# Optional OpenAI SDK
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


def format_time(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def generate_chapters_with_ai(
    segments: list[Segment], *, client: Union[OpenAI, None] = None
) -> list[Chapter]:
    """Generate chapters using OpenAI based on the full video transcript."""
    if not segments or not client:
        logger.warning("No segments or OpenAI client provided for AI chapter generation")
        return []

    # Extract all text with timing information
    full_text = " ".join(seg.text.strip() for seg in segments if seg.text.strip())
    total_duration = segments[-1].end if segments else 0

    try:
        system = """
                    You are a senior YouTube editor. You receive an array of transcript segments:
                    [
                    { "start": <seconds>, "end": <seconds>, "text": "<text>" },
                    ...
                    ]

                    TASK
                    Create human-friendly YouTube CHAPTERS that reflect topic-level sections, not raw sentence fragments.

                    OUTPUT
                    Return ONLY a JSON array (no prose) of objects:
                    [
                    { "start_time": "MM:SS", "title": "Title (<= 12 0 chars)" }
                    ]
                    - If duration >= 1 hour, use "HH:MM:SS", otherwise "MM:SS".
                    - The very first chapter MUST start at "00:00".
                    - Times must be zero-padded, strictly increasing, and < total duration.
                    - Titles must be descriptive, human-readable, sentence case (capitalize first letter), no emojis/hashtags/trailing punctuation.

                    CHAPTER COUNT
                    - ~5 minutes: target 10–12 chapters (acceptable 8–14 if content naturally groups).
                    - Avoid chapters shorter than ~20s unless it’s a clearly distinct topic.

                    BOUNDARY HEURISTICS
                    - Prefer boundaries where a new section/topic/menu is introduced. Look for cues like:
                    "Next", "The first is", "In this section", "Menu", "Components", "Models", "Integrations",
                    "Publication", "Import/Export", "Maintenance", "Localizations", "Extensions", "Templates",
                    "SMTP", "Access", "Permissions", "Roles", "Users", "File storage".
                    - Prefer ends at strong punctuation (., ?, !) or larger pauses (gap between segments).
                    - NEVER split mid-sentence or on dangling fragments such as: "We'll", "you'll", "So", "Also", "Then", "And", "The first is".
                    - When multiple consecutive segments describe the same feature, MERGE them into one chapter.

                    TITLE RULES
                    - Remove filler starts ("So", "Also", "Then", "Now", "Okay").
                    - Use concise, specific titles that summarize the topic (e.g., "Components: list and management").
                    - If two titles would be identical, add a short clarifier.
                    - Keep titles <= 120 chars.

                    TIMING RULES
                    - Compute start_time from the first segment that truly starts a NEW topic.
                    - Round to the nearest second; format as zero-padded MM:SS (or HH:MM:SS).

                    VALIDATION BEFORE RETURN
                    1) First chapter is "00:00".
                    2) Times strictly increase and remain within total duration.
                    3) No title is a dangling fragment; each title <= 120 chars.
                    4) Output is STRICT JSON array with no comments or trailing commas.
                """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": f"Create chapters for this video transcript (duration: {format_time(total_duration)}):\n\n{full_text}",
                },
            ],
            temperature=0.3,
            max_tokens=1000,
        )

        # Parse the JSON response
        import json
        chapters_data = json.loads(response.choices[0].message.content.strip())

        chapters = []
        for chapter_data in chapters_data:
            if "start_time" in chapter_data and "title" in chapter_data:
                chapters.append(Chapter(
                    start_time=chapter_data["start_time"],
                    title=chapter_data["title"]
                ))

        logger.info("Generated %d chapters using AI", len(chapters))
        return chapters

    except Exception as e:
        logger.warning("AI chapter generation failed: %s", e)
        return []


def summarize_description(
    segments: list[Segment],
    *,
    title: Union[str, None] = None,
    use_gpt: bool = False,
    client: Union[OpenAI, None] = None,
) -> str:
    """Generate description summary."""
    if not segments:
        return "No content available."

    # Extract all text
    full_text = " ".join(seg.text.strip() for seg in segments if seg.text.strip())

    if use_gpt and client:
        try:
            system = (
                "You are a YouTube content creator. Create an engaging video description "
                "based on the transcript. Include a brief summary and key takeaways. "
                "Keep it concise and engaging."
            )
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system},
                    {
                        "role": "user",
                        "content": f"Create a YouTube description for this content:\n\n{full_text[:2000]}",
                    },
                ],
                temperature=0.7,
                max_tokens=500,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning("GPT description generation failed: %s", e)

    # Fallback: heuristic summary
    sentences = full_text.split(".")[:3]  # Take first 3 sentences
    summary = ". ".join(sentences).strip()
    if not summary.endswith("."):
        summary += "."

    return summary


def write_youtube_assets(
    outdir: str, title: str, description_md: str, chapters: list[Chapter]
) -> None:
    """Write YouTube assets to files."""
    Path(outdir).mkdir(parents=True, exist_ok=True)

    # Write description
    content = f"# {title}\n\n{description_md}\n\n"
    if chapters:
        content += "## Chapters\n\n"
        for chapter in chapters:
            content += f"{chapter.start_time} - {chapter.title}\n"
    (Path(outdir) / "description.md").write_text(content, encoding="utf-8")

    # Write chapters file
    if chapters:
        chapters_content = "\n".join(f"{chapter.start_time} {chapter.title}" for chapter in chapters)
    else:
        chapters_content = "00:00:00 - Start\n"
    (Path(outdir) / "chapters.txt").write_text(chapters_content, encoding="utf-8")

    logger.info("YouTube assets written to %s", outdir)


def build_chapters_from_sentences(
    sentence_groups: list, segments: list[Segment]
) -> list[Chapter]:
    """Build chapters from sentence groups."""
    chapters = []
    for group in sentence_groups:
        if not group or not group.get("windows"):
            continue
        # Get the start time from the first window
        start_time = group["windows"][0]["start"]
        # Use the text from the group
        text = group.get("text", "").strip()
        # Create title from first sentence, trimmed and capitalized
        title = text.split(".")[0].strip()
        max_title_length = 120
        if len(title) > max_title_length:
            title = title[:117] + "..."
        title = title.capitalize()
        chapters.append(Chapter(start_time=format_time(start_time), title=title))

    # Merge chapters that are too short (less than 20 seconds)
    merged_chapters = []
    for i, chapter in enumerate(chapters):
        if i > 0 and merged_chapters:
            prev_chapter = merged_chapters[-1]
            # If current chapter is too short, merge with previous
            current_start = sum(int(x) * (60 ** (2 - i)) for i, x in enumerate(chapter.start_time.split(":")))
            prev_start = sum(int(x) * (60 ** (2 - i)) for i, x in enumerate(prev_chapter.start_time.split(":")))
            if current_start - prev_start < 20:
                merged_chapters[-1] = Chapter(
                    start_time=prev_chapter.start_time,
                    title=f"{prev_chapter.title} & {chapter.title}"
                )
                continue
        merged_chapters.append(chapter)

    return merged_chapters


def build_chapters_from_srt(segments: list[Segment], gap_thresh: float = 1.5) -> list[Chapter]:
    """Build chapters from SRT segments based on gap threshold."""
    if not segments:
        return []

    chapters = []
    current_start = segments[0].start
    current_text_parts = [segments[0].text.strip()]

    for i in range(1, len(segments)):
        gap = segments[i].start - segments[i-1].end
        if gap > gap_thresh:
            # Gap is large enough, create a chapter
            title = " ".join(current_text_parts).split(".")[0].strip()
            max_title_length = 120
            if len(title) > max_title_length:
                title = title[:117] + "..."
            title = title.capitalize()
            chapters.append(Chapter(start_time=format_time(current_start), title=title))

            # Start new chapter
            current_start = segments[i].start
            current_text_parts = [segments[i].text.strip()]
        else:
            current_text_parts.append(segments[i].text.strip())

    # Add the last chapter
    if current_text_parts:
        title = " ".join(current_text_parts).split(".")[0].strip()
        max_title_length = 120
        if len(title) > max_title_length:
            title = title[:117] + "..."
        title = title.capitalize()
        chapters.append(Chapter(start_time=format_time(current_start), title=title))

    return chapters


def build_chapters_from_blocks(blocks: list, segments: list[Segment]) -> list[Chapter]:
    """Build chapters from blocks."""
    chapters = []
    for block in blocks:
        if not block.get("segments"):
            continue
        start_time = block["segments"][0]["start"]
        text = " ".join(seg["text"].strip() for seg in block["segments"] if seg["text"].strip())
        title = text.split(".")[0].strip()
        max_title_length = 120
        if len(title) > max_title_length:
            title = title[:117] + "..."
        title = title.capitalize()
        chapters.append(Chapter(start_time=format_time(start_time), title=title))

    return chapters


def generate_youtube_assets(
    workdir: str,
    youtube_source: str = "sentences",
    gap_thresh: float = 1.5,
    *, use_ai: bool = True,
    client: Union[OpenAI, None] = None,
    title_override: Union[str, None] = None,
) -> YouTubeAssets:
    """Generate complete YouTube assets."""
    logger.info(f"Generating YouTube assets: workdir={workdir}, use_ai={use_ai}, client={client is not None}")

    # Load segments
    segments_path = Path(workdir) / "subs.srt"
    if not segments_path.exists():
        msg = f"SRT file not found: {segments_path}"
        raise FileNotFoundError(msg)

    from .srt_utils import parse_srt

    segments = parse_srt(str(segments_path))
    logger.info(f"Loaded {len(segments)} segments from SRT")

    # Generate chapters using AI by default
    if use_ai and client:
        chapters = generate_chapters_with_ai(segments, client=client)
        if not chapters:
            logger.warning("AI chapter generation failed, falling back to SRT-based chapters")
            chapters = build_chapters_from_srt(segments, gap_thresh)
    else:
        # Fallback to SRT-based chapters if AI is not available
        logger.info("Using SRT-based chapter generation (AI client not available)")
        chapters = build_chapters_from_srt(segments, gap_thresh)

    # Generate title
    title = title_override or f"Video Content - {format_time(segments[-1].end if segments else 0)}"

    # Generate description
    description = summarize_description(segments, title=title, use_gpt=use_ai, client=client)

    return YouTubeAssets(title=title, description=description, chapters=chapters)

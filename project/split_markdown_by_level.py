import re


def split_markdown_by_level(text, level=2):
    """
    Split markdown text into sections starting at a given header level.

    Each returned section includes its matching header and all following
    content until the next header of the same level.

    Content before the first matching header is preserved as its own section
    if it is non-empty.
    """
    if level < 1 or level > 6:
        raise ValueError("level must be between 1 and 6")

    pattern = re.compile(
        rf'^\s*(#{{{level}}})\s+(.+?)\s*#*\s*$',
        re.MULTILINE
    )

    matches = list(pattern.finditer(text))
    if not matches:
        return [text.strip()] if text.strip() else []

    sections = []

    # Preserve text before the first matching header
    first_start = matches[0].start()
    intro = text[:first_start].strip()
    if intro:
        sections.append(intro)

    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        section = text[start:end].strip()
        if section:
            sections.append(section)

    return sections

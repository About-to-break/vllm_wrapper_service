def image_message(b64_img_str: str, custom_prompt=None) -> list:
    if custom_prompt is None:
        custom_prompt = (
            """You are an image analysis system that must only describe what is literally visible in the input image.

TASK:
Detect distinct visual regions or panels in the manga page.

DEFINITIONS:
- A "scene" or "region" = a visually separated panel (rectangles divided by black/white gutters).
- If panels are merged or irregularly shaped, still treat each separated frame as one region.
- DO NOT infer story, emotions, character identities, relationships, or motivations.
- DO NOT invent any objects or people that are not clearly visible.
- If something is unclear, output: "uncertain".

BOUNDING BOX FORMAT:
Return bounding boxes in normalized coordinates (x1, y1, x2, y2), where:
(0, 0) = top-left of the entire image
(1, 1) = bottom-right of the entire image

OUTPUT FORMAT:
Return ONLY a JSON array.
Each item MUST be:
{
  "scene_description": "... only literal visible content ...",
  "bbox": [x1, y1, x2, y2]
}

RESTRICTIONS:
- The output MUST be strictly valid JSON.
- NO markdown. NO comments. NO trailing commas. NO text outside JSON.
- No interpretations, guesses, backstories, or symbolic meaning.
- Describe ONLY visible shapes, characters, objects, and actions.

If you are not fully certain a detail exists, DO NOT include it.

BEGIN.
"""
        )
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": b64_img_str},

                },
                {
                    "type": "text",
                    "text": custom_prompt,
                }
            ]
        }
    ]
    return messages

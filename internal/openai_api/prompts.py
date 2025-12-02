def image_message(b64_img_str: str, custom_prompt=None) -> list:
    if custom_prompt is None:
        custom_prompt = (
            "Analyze the manga image and detect all distinct scenes or major visual regions (e.g., a person, a car, a building, a landscape area, a scene or frame). "
            "For each detected scene or region, provide a bounding box in normalized coordinates (x1, y1, x2, y2), where (0,0) is top-left and (1,1) is bottom-right. "
            "Return the result as a JSON list of objects with keys 'scene_description' and 'bbox'. "
            "Do not include any other text. Only output strict valid JSON."
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

import json

from embedding.chunk_builder import build_chunks_from_data, build_h2h_chunks_from_data


def test_memgallery_profile_can_be_excluded_from_chunk_text():
    payload = {
        "character_profile": {"name": "Alice", "persona_summary": "A photographer."},
        "multi_session_dialogues": [
            {
                "session_id": "D1",
                "date": "2025-01-01",
                "dialogues": [
                    {"round": "D1:1", "user": "Hello", "assistant": "Hi"}
                ],
            }
        ],
    }
    with_profile = build_chunks_from_data(payload, ".", "sample")
    without_profile = build_chunks_from_data(
        payload, ".", "sample", include_profile=False
    )

    assert "profile_summary:" in with_profile[0].text
    assert "A photographer." in with_profile[0].text
    assert "profile_summary:" not in without_profile[0].text
    assert without_profile[0].metadata["profile_name"] == "Alice"


def test_h2hmem_caption_is_loaded_beside_the_session_image(tmp_path):
    session_dir = tmp_path / "dyadic" / "dialogue1" / "scenes" / "session1"
    caption_dir = session_dir / "caption"
    caption_dir.mkdir(parents=True)
    (caption_dir / "1.json").write_text(
        json.dumps(
            {
                "success": True,
                "description": {"final_text": "A cat rests on a pink blanket."},
            }
        ),
        encoding="utf-8",
    )
    session = {
        "session_id": "session1",
        "timeline_date": "2025-01-01",
        "dialogue": [
            {
                "role": "Alice",
                "content": {"text": "Look at Almond.", "image": ["1.png"]},
            },
            {"role": "Bob", "content": {"text": "She looks relaxed."}},
        ],
    }

    with_captions = build_h2h_chunks_from_data(
        session,
        session_path=session_dir / "session.json",
        variant="dyadic",
        conversation_id="dialogue1",
    )
    without_captions = build_h2h_chunks_from_data(
        session,
        session_path=session_dir / "session.json",
        variant="dyadic",
        conversation_id="dialogue1",
        include_captions=False,
    )

    assert "image_caption: A cat rests on a pink blanket." in with_captions[0].text
    assert with_captions[0].metadata["image_captions"] == [
        "A cat rests on a pink blanket."
    ]
    assert "image_caption:" not in without_captions[0].text
    assert without_captions[0].metadata["image_captions"] == []

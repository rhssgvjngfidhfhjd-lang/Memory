# Copyright (2025) Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import pprint
import pickle
import argparse
from mmagent.retrieve import translate
   
ONLY       = None    # "episodic" / "semantic" / None(all)
MAX_LEN    = None    # How many characters are displayed in one line, None for no truncation
SHOW_FACES = True    # False if there is no graphical environment

def truncate(text: str, max_len: int | None) -> str:
    if not max_len or len(text) <= max_len:
        return text
    return text[:max_len] + "…"

def print_clip_full(vg, clip_id: int,
                    only: str | None = None,
                    max_len: int | None = None,
                    show_faces: bool = True) -> None:
    # 1) ---------- Text Memory ----------
    node_ids = vg.text_nodes_by_clip.get(clip_id)
    if not node_ids:
        print(f"[Warning] clip_id={clip_id} does not exist or the clip does not have a text node")
        return

    print(f"\n======= Clip {clip_id} Memory =======")
    for nid in node_ids:
        node = vg.nodes[nid]
        if only and node.type != only:
            continue
        contents = node.metadata.get("contents", [])
        contents = translate(vg, contents)
        contents = [truncate(c, max_len) for c in contents]
        print(f"[{node.type:^8}] id={nid:<4} | " +
              pprint.pformat(contents, compact=True))

    # 2) ---------- Related Face / Voice ----------
    face_nodes, voice_nodes = set(), set()
    for nid in node_ids:
        face_nodes.update(vg.get_connected_nodes(nid, type=['img']))
        voice_nodes.update(vg.get_connected_nodes(nid, type=['voice']))

    face_nodes, voice_nodes = sorted(face_nodes), sorted(voice_nodes)

    # ---- Faces
    if face_nodes:
        print(f"\n======= Clip {clip_id} Face ({len(face_nodes)} face nodes in total) =======")
        if show_faces:
            vg.print_faces(face_nodes, print_num=3)
        else:
            for fid in face_nodes:
                imgs = vg.nodes[fid].metadata.get("contents", [])
                print(f"[face] id={fid:<4} | face_num={len(imgs)} "
                      f"| base64: {imgs[0][:50]+'…' if imgs else 'N/A'}")
    else:
        print("\n(no related face)")

    # ---- Voices
    if voice_nodes:
        print(f"\n======= Clip {clip_id} Voice ({len(voice_nodes)} voice nodes in total) =======")
        for vid in voice_nodes:
            audios = vg.nodes[vid].metadata.get("contents", [])
            print(f"[voice] id={vid:<4} | voice_num={len(audios)} "
                  f"| Content: {truncate(str(audios[0]), max_len) if audios else 'N/A'}")
    else:
        print("\n(no related voice)")

# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mem_path", type=str, default="data/memory_graphs/robot/bedroom_01.pkl")
    parser.add_argument("--clip_id", type=int, default=0)
    args = parser.parse_args()

    # --------------------------------------------------------------------------- #
    with open(args.mem_path, "rb") as f:
        graph = pickle.load(f)
        graph.refresh_equivalences()

    print_clip_full(graph,
        clip_id=args.clip_id,
        only=ONLY,
        max_len=MAX_LEN,
        show_faces=SHOW_FACES
    )

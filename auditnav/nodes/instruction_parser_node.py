#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# instruction_parser_node.py
# Instruction Parser Node — translates free-form NL instructions to structured YOLO queries.

# commander.py
# Semantic Navigation Commander - LLM Automated Middleware

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import requests
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_loader import load_config, get
import json
import os
import re

CONFIG = {
    "API_KEY": os.environ.get("SILICONFLOW_API_KEY", "your_api_key_here"),
    "API_URL": "https://api.siliconflow.cn/v1/chat/completions",
    "API_MODEL": "Qwen/Qwen2.5-72B-Instruct",
    "VOCAB_FILE": "yolo_world_vocab_for_llm.json"
}

class InstructionParserNode(Node):
    def __init__(self):
        super().__init__('instruction_parser_node')

        # Load centralised config
        self._cfg = load_config()
        global CONFIG
        CONFIG = {
            'API_KEY': self._cfg['api']['key'],
            'API_URL': get(self._cfg, 'api', 'url', default='https://api.siliconflow.cn/v1/chat/completions'),
            'API_MODEL': get(self._cfg, 'api', 'llm_model', default='Qwen/Qwen2.5-72B-Instruct'),
        }

        # 1. Subscribe to raw instruction from evaluation node
        self.sub_raw = self.create_subscription(String, '/audit_nav/raw_instruction', self.raw_instruction_cb, 10)

        # 2. Publish optimized instruction to navigation system
        self.pub_opt = self.create_publisher(String, '/audit_nav/instruction', 10)

        self.vocab_context = self.load_vocabulary()

        self.get_logger().info("\n" + "="*60)
        self.get_logger().info("Smart Semantic Navigation Commander (LLM Automated Middleware) ready")
        self.get_logger().info("Listening: /audit_nav/raw_instruction -> Publishing: /audit_nav/instruction")
        self.get_logger().info("="*60 + "\n")

    def load_vocabulary(self):
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            file_path = os.path.join(current_dir, CONFIG["VOCAB_FILE"])
            if not os.path.exists(file_path):
                file_path = CONFIG["VOCAB_FILE"]

            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.get_logger().info(f"Loaded vocabulary: {len(data.get('vocabularies', {}).get('COCO_80', []))} COCO classes")
                return data
        except Exception as e:
            self.get_logger().error(f"Cannot load vocabulary: {e}")
            return None

    def optimize_prompt_with_llm(self, user_input):
        self.get_logger().info(f"Calling LLM to optimize instruction: '{user_input}'...")
        vocab_str = json.dumps(self.vocab_context) if self.vocab_context else "No vocabulary provided."

        system_prompt = f"""
        Role: You are an Open-Vocabulary Semantic Extractor for YOLO-World.

        Objective:
        Extract the core target object from the user's input and translate it into a precise English noun phrase.
        You MUST preserve the specific visual and functional attributes of the object.

        Reference Vocabulary:
        {vocab_str}

        CRITICAL MAPPING RULES (Priority Order):

        1. **DROP SPATIAL CONTEXT (Drop)**:
           Completely remove all room names, background references, and spatial prepositions.

        2. **PRESERVE OBJECT MODIFIERS (Keep)**:
           Do NOT over-generalize to broad categories. Retain modifiers that describe the object's
           specific visual appearance or function.
           - [Function]: "desk lamp" -> ["desk lamp"], "office chair" -> ["office chair"]
           - [Color/Material]: "wooden coffee table" -> ["wooden coffee table"]

        3. **STRICT COCO MAPPING (Only for exact synonyms)**:
           Only map to a COCO class if the user's target is a DIRECT synonym.
           - "sofa" -> ["couch"] (Valid synonym mapping)
           - "red sofa" -> ["red couch"] (Keep color modifier)

        4. **FORMAT**: Output a JSON list of strings ONLY (e.g., ["wooden desk lamp"]).
        """

        payload = {
            "model": CONFIG["API_MODEL"],
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ],
            "stream": False,
            f"temperature": get(self._cfg, "commander", "llm_temperature", default=0.1),
            "max_tokens": get(self._cfg, "commander", "llm_max_tokens", default=100)
        }

        headers = {
            "Authorization": f"Bearer {CONFIG['API_KEY']}",
            "Content-Type": "application/json"
        }

        try:
            response = requests.post(CONFIG["API_URL"], json=payload, headers=headers, timeout=10)
            response.raise_for_status()
            result = response.json()

            content = result['choices'][0]['message']['content'].strip()

            match = re.search(r'\[.*?\]', content, re.DOTALL)

            if match:
                clean_json_str = match.group(0)
            else:
                clean_json_str = content.replace('```json', '').replace('```', '').strip()

            try:
                optimized_list = json.loads(clean_json_str)
                if isinstance(optimized_list, list) and len(optimized_list) > 0:
                    return str(optimized_list[0])
            except json.JSONDecodeError:
                return clean_json_str.strip('"\'')

            return content

        except Exception as e:
            self.get_logger().error(f"LLM call failed: {e}")
            return user_input

    def raw_instruction_cb(self, msg: String):
        """Triggered when evaluation node sends a raw instruction."""
        raw_cmd = str(msg.data).strip()
        if not raw_cmd:
            return

        optimized_cmd = self.optimize_prompt_with_llm(raw_cmd)

        self.get_logger().info(f"   -> Raw input:      \"{raw_cmd}\"")
        self.get_logger().info(f"   -> Optimized:      \"{optimized_cmd}\"")

        out_msg = String()
        out_msg.data = optimized_cmd
        self.pub_opt.publish(out_msg)


def main(args=None):
    rclpy.init(args=args)
    node = InstructionParserNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

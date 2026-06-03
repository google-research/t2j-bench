# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Mock reward server for PPO training testing.

Returns random reward scores for any input messages.
"""

import json
import random
from http.server import HTTPServer, BaseHTTPRequestHandler


class RewardHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        request = json.loads(post_data.decode('utf-8'))

        messages = request.get("messages", [])
        # Return random scores between -1 and 1
        scores = [random.uniform(-1.0, 1.0) for _ in messages]

        response = json.dumps({"scores": scores})
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(response.encode('utf-8'))

    def log_message(self, format, *args):
        # Suppress default logging
        pass


if __name__ == "__main__":
    server = HTTPServer(('localhost', 8000), RewardHandler)
    print("Mock reward server started on http://localhost:8000")
    server.serve_forever()

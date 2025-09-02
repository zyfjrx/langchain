import json

import requests


class DifyApiUtils:
    base_url = ''
    headers = {}
    data = {}

    def __init__(self, base_url: str = None, headers: dict = None):

        if base_url is None:
            # 读取配置
            config_base_url = "http://14.103.162.45/v1/"
            if config_base_url is None or config_base_url == "":
                config_base_url = "http://localhost/v1"
            self.base_url = config_base_url
        else:
            self.base_url = base_url
        # 读取配置
        secret_key = "app-FhRnm20OJoMJ1qcbsPbTLfjG"
        if headers is None:
            self.headers = {
                'Authorization': 'Bearer {0}'.format(secret_key),
                'Content-Type': 'application/json',
            }

    def chat_message(self, query: str = None):
        chat_message_url = "{0}/{1}".format(self.base_url, "chat-messages")
        self.data = {
            "inputs": {
            },
            "query": "{0}".format(query),
            "response_mode": "streaming",
            "user": "zyf",
            "conversation_id": "",
            "files": [],
        }
        response = requests.post(chat_message_url, headers=self.headers, data=json.dumps(self.data))
        if response.status_code == 200:
            res = self.response_deal_message(response_content=response.content)
            return res
        else:
            return False

    def response_deal_message(self, response_content: bytes):
        res_list = response_content.decode(encoding="utf-8").split("\n\n")
        result = ""
        for res in res_list:
            temp_list = res.split(":", 1)
            if temp_list[0] == "data":
                json_str = temp_list[1]
                json_obj = self.event_deal_message(json_str)
                if json_obj['event'] == 'message':
                    answer = json_obj['answer']
                    result += answer
        return result



    def event_deal_message(self, json_str: str):
        # print(json_str)
        event_data = json.loads(json_str)
        return event_data


if __name__ == "__main__":

    dify_client = DifyApiUtils()
    inputs = "阿比西尼亚猫的生活环境要求是什么？"
    result = dify_client.chat_message(query=inputs)

    print(result)
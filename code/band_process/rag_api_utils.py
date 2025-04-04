"""
每天新建一个知识库
"""

import random
import string
import time
import traceback

import requests
from loguru import logger

OPENAI_API_BASE_URL = "http://192.168.31.27:3000/api"
OPENAI_API_KEY = "fastgpt-qI28UcpMLOQ2sZf6fKB6zT4QWFYsBBLyCbrIV7kTTdhFWONEhiYu1Cxsf2LLmiJ"


def create_dataset(dataset_name):
    """
    创建知识库
    :return:
    """
    create_dataset_url = rf'{OPENAI_API_BASE_URL}/core/dataset/create'
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {OPENAI_API_KEY}"}
    data = {
        "type": "dataset",
        "name": dataset_name,
        "vectorModel": "m3e-base",
        "agentModel": "hwen2.5-instruct"
    }
    result = requests.post(url=create_dataset_url, json=data, headers=headers)
    print(result.json())


def query_dataset():
    """
    查询知识库
    :return:
    """
    dataset_dict = {}
    query_dataset_url = rf'{OPENAI_API_BASE_URL}/core/dataset/list'
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {OPENAI_API_KEY}"}
    result = requests.post(url=query_dataset_url, headers=headers)
    try:
        ret = result.json()
        # 按照name-id解析知识库
        for data in ret.get('data', []):
            dataset_dict[data.get('name')] = {
                "id": data.get('_id'),
                "vectorModel": data.get('vectorModel')
            }
    except Exception as e:
        logger.error(f"get token failed: {e}, {result} {result.json()}")
        traceback.print_exc()
    return dataset_dict


def create_text_collection(dataset_name, dataset, text):
    """
    创建纯文本集合
    :return:
    """
    text_collection_url = rf'{OPENAI_API_BASE_URL}/core/dataset/collection/create/text'
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {OPENAI_API_KEY}"}
    data = {
        "text": text,
        "datasetId": dataset.get('id'),
        "name": dataset_name,
        "trainingType": "chunk"
    }
    result = requests.post(url=text_collection_url, json=data, headers=headers)
    # print(result.json())


def chat(message, c_time, chat_id, share_id):
    """
    查询知识库
    :return:
    """
    chat_url = rf'{OPENAI_API_BASE_URL}/v1/chat/completions'
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {OPENAI_API_KEY}"}
    data = {
        "messages": message,
        "variables": {"cTime": c_time},
        "shareId": share_id,
        "chatId": chat_id,
        "detail": True,
        "stream": False,
    }
    logger.info(f"{chat_url} {data}")
    result = requests.post(url=chat_url, headers=headers, json=data)
    try:
        ret = result.json()
        logger.info(f"{ret}")
    except Exception as e:
        logger.error(f"chat failed: {e}, {result} {result.text}")
        traceback.print_exc()
        return None
    return ret


def get_code(k):
    return ''.join(random.sample(string.ascii_letters + string.digits, k))


if __name__ == '__main__':
    # dataset_dict = query_dataset()
    # dataset_name = r'舆情知识库实时'
    # if dataset_name not in dataset_dict:
    #     create_dataset(dataset_name)
    # print(dataset_dict[dataset_name])
    # create_text_collection(dataset_name, dataset_dict[dataset_name], "67868768768787")
    message = [
        {
            "dataId": get_code(32),
            "role": "user",
            "content": "根据知识库描述：一个病人的艾滋病检测条带的结果如下：该检测条带上包含'Baseline'、'gp160'、 'gp120'、 'p66'、 'p51/p55'、 'gp41'、 'p31'、 'p24'、 'p17'、 'Control'等点位，那么这个病人处于艾滋病的什么阶段"
        }
    ]
    current_time = time.time()
    c_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(current_time))
    chat_id = get_code(8)
    # share_id = "7eg9zipoetjpdehgmkftqu25"
    share_id = "x1i9ji7iwery2p9jorcxkby4"
    # out_link_uid = "shareChat-1728302665412-x4D86dGprtt23IL9T7rlj8Iu"
    out_link_uid = "shareChat-1730009285348-_qFSOP0aZcd6lYkhMTZr7o8m"
    result = chat(message, c_time, chat_id, share_id, out_link_uid)
    logger.info(f"{result['choices'][0]['message']['content']}")



    # print(dataset_dict)

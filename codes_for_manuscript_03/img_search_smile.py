# coding: utf-8
# 开发团队：重大化工LC214 AI小分队
# 开发人员：Tristan
# 开发时间：2023/2/22—21:07
# 文件名称：try01 py
# 开发工具：PyCharm

import time
import os
import json

import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC


def upload(img_path):
    select_xpath = '/html/body/center/form/table/tbody/tr[3]/td[1]/input[2]'
    submit_xpath = '//*[@id="b_upload"]'
    clear_xpath = '/html/body/center/form/table/tbody/tr[3]/td[1]/center/input[2]'

    chrome.find_element_by_xpath(clear_xpath).click()
    chrome.find_element_by_xpath(select_xpath).send_keys(img_path)
    submit_button = chrome.find_element_by_xpath(submit_xpath).click()


def get_information():
    get_smiles_xpath = '//*[@id="b_getsmiles"]'
    smiles_xpath = '/html/body/center/form/table/tbody/tr[3]/td[2]/input[1]'

    chrome.find_element_by_xpath(get_smiles_xpath).click()
    text = chrome.find_element_by_xpath(smiles_xpath).get_attribute("value")

    return text


def img2smiles(chromedriver_path=r'D:\Tristan\Anaconda3\envs\awen\Lib\site-packages\chromedriver.exe',
         img_folder='D:\\Desktop\\mol_img', save=True):
    global chrome
    # 不开网页搜索
    # chromedriver_path = r'D:\Tristan\Anaconda3\envs\awen\Lib\site-packages\chromedriver.exe'
    option = webdriver.ChromeOptions()
    option.add_argument("headless")
    chrome = webdriver.Chrome(executable_path=chromedriver_path, chrome_options=option)
    chrome.get('https://cactus.nci.nih.gov/cgi-bin/osra/index.cgi')
    wait = WebDriverWait(chrome, 20)

    # img_folder = 'D:\\Desktop\\mol_img'
    imgs76 = os.listdir(img_folder)
    smiles_list = {}
    for img in imgs76:
        print('**')
        upload(img_folder.replace('/', '\\') + '\\' + img)
        time.sleep(7)
        try:
            tmp_text = get_information()
            chrome.save_screenshot('res/' + img.rstrip('.jpg') + '.png')
            smiles_list[img] = [tmp_text]
        except:
            smiles_list[img] = ['Sorry, no structures found']

    print(smiles_list)
    chrome.quit()
    if save:
        smiles_list = pd.DataFrame(smiles_list)
        smiles_list.to_csv(img_folder+"\\smiles_list.csv")
    # with open('result.json', 'w') as fp:
    #     json.dump(smiles_list, fp)


if __name__ == '__main__':
    img2smiles()
# 110.0.5481.104
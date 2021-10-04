import requests
# 引用BeautifulSoup库
from bs4 import BeautifulSoup
import csv
# 调用open()函数打开csv文件，传入参数：文件名“articles.csv”、写入模式“w”、newline=''。
csv_file=open('whether2.csv','w',newline='',encoding='utf-8-sig')
# 用csv.writer()函数创建一个writer对象。
writer = csv.writer(csv_file)
list2=['Time','High_temperature','low_temperature']
# 调用writer对象的writerow()方法，可以在csv文件里写入一行文字 “标题”和“链接”和"摘要"。
writer.writerow(list2)

city = 'jiading'

years = ['2021']

months = ['01','02','03']

for year in years:
    for month in months:
        url = 'http://lishi.tianqi.com/' + city + '/' + str(year) + str(month) + '.html'
        print (url)
        headers = {'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36'}
        res_parameter = requests.get(url,headers=headers)
        bs_parameter = BeautifulSoup(res_parameter.text,'html.parser')
        tag_name = bs_parameter.find('ul', class_='thrui')
        parameter_names =  tag_name.find_all('li')
        for parameter_name in parameter_names:
            time = parameter_name.find(class_='th200').text
            Data = parameter_name.find_all(class_='th140')
            list1 = []
            for data in Data:
                list1.append(data.text)
            High_temperature = list1[0]
            low_temperature = list1[1]
            list3 = [time,High_temperature, low_temperature]
            writer.writerow(list3)
csv_file.close()






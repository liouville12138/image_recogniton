from bs4 import BeautifulSoup
import requests
 
 
if __name__=='__main__':
    url='http://192.168.1.11:8888/test.html'
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.87 Safari/537.36"}
    req = requests.get(url=url, headers=headers)
    req=requests.get(url=url,headers=headers)
    req.encoding = 'gb2312'
    html=req.text
    bf=BeautifulSoup(html,'html.parser')
    targets_url=bf.find('img')['src']
    name="cat" + "test" +'.jpg'
    path='D:\\learning\\ai\\cat'
    file_name = path + '\\' + name
    try:
        req1=requests.get(targets_url,headers=headers)
        f=open(file_name,'wb')
        f.write(req1.content)
        f.close()
    except:
        print("some error")
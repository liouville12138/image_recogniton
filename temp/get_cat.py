from bs4 import BeautifulSoup
import requests
 
 

def get_cat(pageIndex, imageCnt):
    url='https://www.veer.com/query/image/?phrase=%E7%8C%AB&page=' + str(pageIndex)
    host = "https://www.veer.com"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.87 Safari/537.36"}
    req = requests.get(url=url, headers=headers)
    req=requests.get(url=url,headers=headers)
    req.encoding = 'utf-8'
    html=req.text
    bf=BeautifulSoup(html,'html.parser')
    targets_url=bf.find('section', class_="assets simplify_search gi_bricks") .find_all('a',target='_blank')
    if targets_url == None :
        return 0
    for each in targets_url:
        each_url = each.get('href')
        if "http" not in each_url:
            each_url = host + each_url
        img_req=requests.get(each_url, headers=headers)
        img_req.encoding = 'utf-8'
        html = img_req.text
        bf = BeautifulSoup(html, 'html.parser')
        img_url = bf.find('div', class_='unzoomed').find('img')['src']
        name="cat" + str(imageCnt) +'.jpg'
        imageCnt += 1
        path='D:\\learning\\ai\\cat'
        file_name = path + '\\' + name
        try:
            req1=requests.get(img_url,headers=headers)
            f=open(file_name,'wb')
            f.write(req1.content)
            f.close()
        except:
            print("some error")
    return imageCnt

if __name__=='__main__':
    pageIndex = 4
    imageCnt = 301
    image_return = imageCnt
    while(image_return != 0):
        image_return = get_cat(pageIndex, imageCnt)
        pageIndex += 1
        imageCnt = image_return

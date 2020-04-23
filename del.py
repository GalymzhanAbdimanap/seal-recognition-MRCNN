import glob 
img_name_arr =[]
for i in ['jpg', 'jpeg', 'png', 'jfif','JPG', 'JPEG', 'PNG']:
	imgs = glob.glob('*.'+i)

	for img in imgs:
		img_name_arr.append(img)

#print(img_name_arr)
import json

with open('via_region_data.json', 'r', encoding='utf-8') as f: #открыли файл
    text = json.load(f) #загнали все из файла в переменную

delete_arr=[]
values = text.keys()
for x in values:
	i = text[x]
	if i['filename'] not in img_name_arr:
		delete_arr.append(x)


#print(delete_arr)

for x in delete_arr:
	del text[x]

with open('via_region_data.json', 'w',  encoding='utf-8') as f:
    text = json.dump(text, f, ensure_ascii=False)








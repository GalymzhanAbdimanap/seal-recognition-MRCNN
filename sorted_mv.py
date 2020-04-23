import json, shutil

with open('via_region_data.json', 'r', encoding='utf-8') as f: #открыли файл
    text = json.load(f) #загнали все из файла в переменную	


values = text.values()

for i in values:
	try:
		shutil.move(i['filename'], 'seal_dataset/'+i['filename'])
	except Exception as e:
		print(i['filename'])
		print(e)
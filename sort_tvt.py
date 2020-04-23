import json, shutil

with open('via_region_data.json', 'r', encoding='utf-8') as f: #открыли файл
    text = json.load(f) #загнали все из файла в переменную	

'''def js_dump(name_json):
	
	with open(name_json, 'r', encoding='utf-8') as f: #открыли файл
	    text = json.load(f) #загнали все из файла в переменную	
	

	with open(name_json, 'w',  encoding='utf-8') as f:
	    text = json.dump(text, f, ensure_ascii=False)'''


values = text.values()
count=0
for i in values:
	#i = text[x]
	try:
		if count<1447:
			shutil.move(i['filename'], 'train/'+i['filename'])
		elif count>=1447 and count<1756:
			shutil.move(i['filename'], 'val/'+i['filename'])
		else:
			shutil.move(i['filename'], 'test/'+i['filename'])
		count+=1
	except Exception as e:
		print(i['filename'])
		print(e)

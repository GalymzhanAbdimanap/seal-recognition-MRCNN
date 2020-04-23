import json

with open('via_export_json.json', 'r', encoding='utf-8') as f: #открыли файл
    text = json.load(f) #загнали все из файла в переменную

values = text.keys()
for x in values:
	if x[:3]!='p2_':
		print(x)
		#i = text[x]
		#new_key = 'p2_'+str(x)
		#newValue = 'p2_' + str(i['filename'])
		#text[new_key] = text.pop(x)
		#print(new_key, newValue)
		#i['filename'] = newValue
	else:
		continue





'''with open('via_export_json.json', 'w',  encoding='utf-8') as f:
    text = json.dump(text, f, ensure_ascii=False)'''

'''import shutil
import glob
for i in ['jfif', 'PNG', 'JPEG', 'JPG']:
	imgs = glob.glob('*.'+i)
	for img in imgs:
		print(img)
		shutil.move(img, 'test/p2_'+img)'''

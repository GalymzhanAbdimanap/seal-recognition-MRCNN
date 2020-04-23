'''import json

with open('via_export_json.json', 'r', encoding='utf-8') as f: #открыли файл
    text = json.load(f) #загнали все из файла в переменную
countDeleteShape=0
countDelete=0
values = text.values()
for i in values:
	#print(i['regions'])
	for ind, j in enumerate(i['regions']):
		print(type(i['regions']))
		for c in j['shape_attributes'].values():
			if str(c)=='rect' or str(c)=='ellipse' or str(c)=='polygon':
				if len(i['regions'])>1:
					#print(len(i['regions']))
					del i['regions'][ind]
					countDeleteShape+=1
					
				else:
					del i
					countDelete+=1
with open('via_export_json.json', 'w') as f:
    text = json.dump(text, f)
					
print("countDeleteShape="+str(countDeleteShape))
print("countDelete="+str(countDelete))'''





import json

with open('via_export_json.json', 'r', encoding='utf-8') as f: #открыли файл
    text = json.load(f) #загнали все из файла в переменную
countDeleteShape=0
countDelete=0
deleteArr=[]
delete=[]
values = text.keys()
for x in values:
	i = text[x]
	#print(i['regions'])
	for ind, j in enumerate(i['regions']):
		#print(type(i['regions']))
		for c in j['shape_attributes'].values():
			if str(c)=='rect' or str(c)=='ellipse' or str(c)=='polygon':
				if len(i['regions'])>1:
					#print(len(i['regions']))
					deleteArr.append(ind)
					#del i['regions'][ind]
					countDeleteShape+=1
					
				else:
					delete.append(x)
					
					countDelete+=1
	if len(deleteArr)>0:
		#print(deleteArr)
		print("-------------------------")
		deleteArr.sort(reverse=True)
		print(deleteArr)
		print("====================================")
		print(i['regions'])
		for k in deleteArr:
			del i['regions'][k]
		deleteArr=[]

with open('via_export_json.json', 'w',  encoding='utf-8') as f:
    text = json.dump(text, f, ensure_ascii=False)



with open('via_export_json.json', 'r', encoding='utf-8') as f: #открыли файл
    text = json.load(f) #загнали все из файла в переменную	
for x in delete:
	del text[x]

with open('via_export_json.json', 'w',  encoding='utf-8') as f:
    text = json.dump(text, f, ensure_ascii=False)

print(delete)

print("countDeleteShape="+str(countDeleteShape))
print("countDelete="+str(countDelete))
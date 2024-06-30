import random
import numpy as np
from collections import defaultdict


def format_filtering(question,class2object,object_index,ranking):
    filter_text = f"Question: {question}\n"
    filter_text += f"Select objects that can help answer the question \n"
    filter_text += "These are the objects available for selection in current scene graph\n"
    for class_name in class2object.keys():
        filter_text += f"{class_name} \n"
    # only require selection when there are more than k objects
    if object_index == 0:
        filter_text += "No object available \n"
    #filter_text += f"Select the top {len(ranking)} important objects\n"
    filter_text += "Rank them based on their importance from high to low \n"
    filter_text += "Reprint the name of objects in ranked order. Each object one line \n"
    filter_text += "If currently no object available, just type 'No object available' \n"
    
    # Jiachen TODO 5: format the filtering answer
    answer = "\n".join(ranking) if object_index > 0 else 'No object available'
    filter_text += "Answer: "
    filter_text += answer + "\n"
    
    return filter_text
    
prefiltering = True
question = "Where is the printer located?"
ranking = ["printer","shelf","table","chair","door"]
#random.shuffle(ranking)
print(ranking)

#obj_map = {str(i):f"obj_{i}" for i in range(20)}
step = ["printer","window","closet","shelf","chair","chair","television","stove"]
for s in step:
    print(s)
obj_map = {c:c for c in step}
predict_size = 10
prediction = np.arange(predict_size)

indices = []
object_classes = []
object_features = []
class2object = defaultdict(list)
object_index = 0
text = ""
top_k_categories = 3

text = f"Question: {question}\n"

text += f"Select the frontier/object that would help finding the answer of the question.\n"
text += "These are the objects already in our scene graph:\n"
for i, sid in enumerate(step):
    if str(sid) not in obj_map.keys():
        continue
    else:
        try:
            object_feature = np.zeros(1024)
            indices.append(i)
            object_classes.append(obj_map[str(sid)])
            object_features.append(object_feature)
            class2object[obj_map[str(sid)]].append(object_index)
            object_index += 1
        except:
            continue
print(indices)
print(object_classes)
print(object_index)
print(class2object)

if prefiltering:
    # 1. filter unseen object categories in ranking
    ranking = [cls for cls in ranking if cls in class2object.keys()]
    print(ranking)
    # 2. take top k object categories
    ranking = ranking[: top_k_categories]
    print(ranking)
    # 3. reformulate the object indices
    indices = [
        indices[obj_idx] for cls in ranking for obj_idx in class2object[cls]
    ]
    print(indices)
    object_classes = [cls for cls in ranking for _ in class2object[cls]]
    print(object_classes)
    object_features = [
        object_features[obj_idx]
        for cls in ranking
        for obj_idx in class2object[cls]
    ]

for i, class_name in enumerate(object_classes):
    text += f"object {i} {class_name} <scene> "

if object_index == 0:
    text += f"No object available "
text += "/\n"

answer = f"object 0"

text += "Below are all the frontiers that we can explore:\n"

prediction = np.concatenate((prediction[indices],prediction[len(step):]))
for i in range(predict_size - len(step)):
    text += f"frontier {i} <scene> "
text += "/\n"

text += f"Answer: {answer}\n"  
print(text)
print(prediction)
print(len(prediction))
print(len(object_classes))
assert len(prediction) == len(object_classes) + (predict_size-len(step)) # this is frontier

print(format_filtering(question,class2object,object_index,ranking))




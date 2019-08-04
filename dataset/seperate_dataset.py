
def count_instances(annotations):
    labels = ["pedestrian", "cyclist", "on-call", "on-mobile", "facing_backwards"]
    stats = {} 
    img_objects = {}
    for line in open(annotations):
        img, meta = line.split(" ")
        x1, x2, y1, y2, class_no = meta.split(",")
        lbl = labels[int(class_no)]
        if lbl not in stats:
            stats[lbl] = 1
        else:
            stats[lbl] += 1
        
        if img not in img_objects:
            img_objects[img] = {"TotalObj": 1, "Labels": [lbl]}
        else:
            img_objects[img]["TotalObj"] += 1
            img_objects[img]["Labels"].append(lbl)
    return stats, img_objects
        
def get_train(annotations, maxTrain=100):
    labels = ["pedestrian", "cyclist", "on-call", "on-mobile", "facing_backwards"]
    train = {}
    for line in open(annotations):
        img, meta = line.split(" ")
        x1, x2, y1, y2, class_no = meta.split(",")
        class_no = class_no.strip()
        if int(class_no) == 4:
            continue

        if class_no not in train:
            train[class_no] = [line]
        else:
            if len(train[class_no]) < 100:
                train[class_no].append(line) # add to training

    return train



if __name__ == "__main__":

    print("[INFO] Parsing annotations")
    stats, img_objects = count_instances("annotations_raw.txt")
    train_annot = get_train("annotations_raw.txt")
    print(stats)

    for t in train_annot:
        print("{0} : {1}".format(t, len(train_annot[t])))

    with open('annotations.txt', 'w') as f:
        for lbl in train_annot:
            for dt in train_annot[lbl]:
                f.write(dt)
    # grouped = [0,0,0,0,0,0,0,0]
    # for img in img_objects:
    #     grouped[img_objects[img]["TotalObj"]] += 1
    # print(grouped)


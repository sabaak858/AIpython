names = []
names.append("saba")
print(names)
names.append("Bhumi")
names.insert(2, "irum")
print(names)
names.remove("saba")
print(names)
otherNames = ["allu", "ninni"]
names.extend(otherNames)
print(names)
what_index = names.index("allu")
print(what_index)
if "saba" in names:
    print("saba found")
    if "Bhumi" not in names:
        print("Bhumi not found")
        names.sort()
        print(names)









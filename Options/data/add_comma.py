file1 = open("first_5_min.csv","r")
file2 = open("new_data.csv","w")

Lines = file1.readlines()
for line in Lines:
    new_line = ","+line 
    file2.writelines(new_line)
file2.close()
file1.close()
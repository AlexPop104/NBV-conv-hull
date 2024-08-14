
import zipfile
import os
import shutil
import random

def get_file_list(input_dir):
    return [file for file in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, file))]

def get_random_files(file_list, N):
    return random.sample(file_list, N)

def copy_files(random_files, input_dir, output_dir):
    for file in random_files:
        #shutil.copytree(os.path.join(input_dir, file), os.path.join(output_dir, file))
        shutil.move(os.path.join(input_dir, file), os.path.join(output_dir, file))

def move_random(input_dir=None, output_dir=None, N=1):
    file_list = get_file_list(input_dir)
    
    if(len(file_list)>=N):
    
        random_files = get_random_files(file_list, N)
        print(str(len(file_list))+" files inside and  " +str(N)+ " random files required") 
        copy_files(random_files, input_dir, output_dir)
        
    else:
        print(str(len(file_list)) +"<" +str(N))

def delete_folders_shutil(path):
    try:
        folder_path = path
        shutil.rmtree(folder_path)
        #print('Folder and its content removed')
    except:
        print('Folder not deleted')

if __name__ == '__main__':


    f = open("/home/alex/Alex_documents/Alex_work_new2/Alex_work/GeoA3/Extra_code_Neurocomputing/Dataset_process/categ_list_shapenet.txt")

    name_shapenet_all = []
    nr_elements_all =[]

    for line in f :
        split_line = line.split(': ')
        name_shapenet_all.append(split_line[0])
        nr_elements_all.append(split_line[1].replace('\n',''))

    dictionary_class_shapenet_all = dict(zip(name_shapenet_all,nr_elements_all))

    f.close()

    f = open("/home/alex/Alex_documents/Alex_work_new2/Alex_work/GeoA3/Extra_code_Neurocomputing/Dataset_process/categ_shapenet.txt")

    id_shapenet = []
    name_shapenet = []

    for line in f :
        split_line = line.split(' ')
        id_shapenet.append(split_line[0])
        name_shapenet.append(split_line[1].replace('\n',''))


    dictionary_class_shapenet = dict(zip(id_shapenet,name_shapenet))

    f.close()

    f = open("/home/alex/Alex_documents/Alex_work_new2/Alex_work/GeoA3/Extra_code_Neurocomputing/Dataset_process/categ_selected_shapenet.txt")


    name_shapenet_selected = []
    nr_elements = []

    for line in f :
        split_line = line.split(': ')
        name_shapenet_selected.append(split_line[0])
        nr_elements.append(split_line[1].replace('\n',''))

    f.close()


    category_list = []

    with zipfile.ZipFile('/mnt/ssd1/Alex_data/Shapenet_all/ShapeNetCore.v1.zip') as archive:
        for file in archive.namelist():
            split_file = file.split('/')
            if len(split_file) == 3:
                if(split_file[1] in id_shapenet):
                    #category_list.append(split_file[1])
                    nr_elements_categ = 0
                    res = [i for i in name_shapenet_all if dictionary_class_shapenet[split_file[1]] in i]
                    nr_elements_categ = int(dictionary_class_shapenet_all[res[0]])
                    if nr_elements_categ > 400:
                    #if nr_elements_categ < 400:
                        category_list.append(split_file[1])



    print(len(category_list))


    input_dir_path = "/mnt/ssd1/Alex_data/Shapenet_all/ShapeNetCore.v1.zip/"
    output_dir_path = "/mnt/ssd1/Alex_data/Cuda_Data/Shapenet_with_image_test_again/"

    nr_files_list = [80 , 20]
    dataset_type_list = ["train" ,"valid"]

    #dataset_type_list = ["train"]
    #nr_files_list = [20]

    for category in (category_list):
       
            with zipfile.ZipFile('/mnt/ssd1/Alex_data/Shapenet_all/ShapeNetCore.v1.zip') as archive:
                for file in archive.namelist():
                    split_file = file.split('/')
                    if len(split_file) >= 3 and (category == split_file[1]):
                        if len(split_file) == 3:
                            print(category+" start")
                        archive.extract(file, '/mnt/cuda_4TB/alex/')
            
            input_dir = os.path.join(input_dir_path,category)

            for j in range(len(nr_files_list)):  
                nr_files = nr_files_list[j]
                dataset_type = dataset_type_list[j] 
                
                output_dir = os.path.join(output_dir_path,dataset_type)

                if ( not os.path.isdir(output_dir)):
                    os.makedirs(output_dir)
                    print(dataset_type+ " folder created")

                output_dir = os.path.join(output_dir,category)

                move_random(input_dir=input_dir, output_dir=output_dir, N=nr_files)


                print(category+" "+dataset_type+" "+str(nr_files) + " done" )
                
            delete_folders_shutil(input_dir)
            print(category+"remaining deleted")
    print("Done")
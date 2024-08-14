
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


    dictionary_class_shapenet = dict(zip(name_shapenet,id_shapenet))

    f.close()

    #category_list = ["airplane","cabinet","car","chair","lamp","sofa","table","watercraft"]

    category_list = ["bus","bed","bookshelf","bench","guitar","motorbike","skateboard","pistol"]

    input_dir_path = "/mnt/ssd1/Alex_data/Test_PCNBV_again/"
    output_dir_path = "/mnt/ssd1/Alex_data/Test_PCNBV_again/Shapenet_with_image/"

    path_archive = '/mnt/ssd1/Alex_data/Shapenet_all/ShapeNetCore.v1.zip'

    if ( not os.path.isdir(input_dir_path)):
        os.makedirs(input_dir_path)
        print("input folder created")

    if ( not os.path.isdir(output_dir_path)):
        os.makedirs(output_dir_path)
        print("output folder created")

    # nr_files_list = [500 , 50]
    # dataset_type_list = ["train" ,"valid"]

    nr_files_list = [50]
    dataset_type_list = ["test"]

    #dataset_type_list = ["train"]
    #nr_files_list = [20]

    for category in (category_list):
            
            print("Starting with "+dictionary_class_shapenet[category]+" category")
       
            with zipfile.ZipFile(path_archive) as archive:

                names_foo = [i for i in archive.namelist() if dictionary_class_shapenet[category] in i ]

                for file in names_foo:
                    archive.extract(file,input_dir_path)

            print(category + " extracted")

            input_dir = os.path.join(input_dir_path,"ShapeNetCore.v1",dictionary_class_shapenet[category])

            for j in range(len(nr_files_list)):  
                nr_files = nr_files_list[j]
                dataset_type = dataset_type_list[j] 
                
                output_dir = os.path.join(output_dir_path,dataset_type)

                if ( not os.path.isdir(output_dir)):
                    os.makedirs(output_dir)
                    print(dataset_type+ " folder created")

                output_dir = os.path.join(output_dir,dictionary_class_shapenet[category])

                move_random(input_dir=input_dir, output_dir=output_dir, N=nr_files)


                print(dictionary_class_shapenet[category]+" "+dataset_type+" "+str(nr_files) + " done" )
                
            delete_folders_shutil(input_dir)
            print(dictionary_class_shapenet[category]+"remaining deleted")
    print("Done")
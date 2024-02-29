def format_req()->None:
    with open("./requirements.txt","r") as f:
        req = f.readlines()
        
    new_req = [dep.split(";")[0]+'\n' for dep in req]
    
    with open("./requirements.txt","w") as f:
        f.writelines(new_req)


if __name__ =='__main__':
    format_req()
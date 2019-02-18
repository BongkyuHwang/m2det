with open("nohup.out", "r") as f:
    loss = 0
    count = 0
    epoch = 1
    for idx, line in enumerate(f.readlines()):
        spl_line = line.split(" ")
        
        if spl_line[0] == 'iter':
            if count !=0 and spl_line[1] == '0':
                print(epoch, count, loss, loss/count)
                epoch += 1
                loss = float(spl_line[5])
                count = 1
            else:
                if spl_line[5] != 'inf':
                    loss += float(spl_line[5])
                    count +=1
    print(epoch, count, loss, loss/count)      

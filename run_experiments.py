from subprocess import call
# import os

# tmp_file = open('result.log',"w")

dropouts = [0.95, 0.9, 0.85,0.8,0.75]
lrs = [0.05, 0.02, 0.01]
lr_index = len(lrs)
dp_index = len(dropouts)


# # run overlapping experiments with hyper-para tunning
# count = 0
# for i in range(lr_index):
#     for j in range(dp_index):
#         count += 1
#         overlap_cmd = "python3 overlapping-community-detection.py --lr {lr} --dropout {dropout}".format(lr = lrs[i], dropout = dropouts[j])
#         overlap_call_list = overlap_cmd.split(" ")
#         call(overlap_call_list)
#         with open("overlapping.log","r") as f:
#             result = f.readlines()[-1]
#             f.close()

#         output = "Exp {i}: lr = {lr}, dropout = {dropout} ".format(i = count, lr = lrs[i], dropout = dropouts[j]) + result
#         with open("overlapping_exp_results.log","a") as f:
#             f.write(output)
#             f.close()



# run nonoverlapping expriments with hyper-para tunning
count = 0
# for i in range(lr_index):
for j in range(dp_index):
    count += 1
    overlap_cmd = "python3 nonoverlapping-community-detection.py --lr {lr} --dropout {dropout}".format(lr = 0.05, dropout = dropouts[j])
    overlap_call_list = overlap_cmd.split(" ")
    call(overlap_call_list)
    with open("nonoverlapping.log","r") as f:
        result = f.readlines()[-1]
        f.close()

    output = "Exp {i}: lr = {lr}, dropout = {dropout} ".format(i = count, lr = 0.05, dropout = dropouts[j]) + result
    with open("nonoverlapping_exp_results.log","a") as f:
        f.write(output)
        f.close()


# count = 0
# eps = [1001]
# eps_index = len(eps)
# for j in range(eps_index):
#     count += 1
#     overlap_cmd = "python3 nonoverlapping-community-detection.py --epochs {ep} --lr 0.05".format(ep = eps[j])
#     overlap_call_list = overlap_cmd.split(" ")
#     call(overlap_call_list)
#     with open("nonoverlapping.log","r") as f:
#         result = f.readlines()[-1]
#         f.close()

#     output = "Exp {i}: epoch = {ep} ".format(i = count, ep = eps[j]) + result
#     with open("nonoverlapping_exp_results.log","a") as f:
#         f.write(output)
#         f.close()



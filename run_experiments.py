from subprocess import call
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--overlap", type=bool, default=False, help="decide to run the overlap experiment or not")
parser.add_argument("--multi_hyper", type=bool, default=False, help="decide to run the experiment with several different hyper parameters or not")


dropouts = [0]
dp_index = len(dropouts)

def run_experiment(overlap,multi_hyper):
    if overlap:
        run_file = "overlapping-community-detection.py"
        log_file = "overlapping.log"
        result_file = "overlapping_exp_results.log"
    else:
        run_file = "nonoverlapping-community-detection.py"
        log_file = "nonoverlapping.log"
        result_file = "nonoverlapping_exp_results.log"

    if not multi_hyper and not overlap:
        cmd = "python3 " + run_file
        call_list = cmd.split(" ")  
        call(call_list)
        with open(log_file, "r") as f:
            result = f.readlines()[-1]
            f.close() 
        output = "Exp {i}: initial_lr = {lr} ".format(i = 0, lr = 0.05) + result
        with open(result_file,"a") as f:
            f.write(output)
            f.close()
    elif not multi_hyper and overlap:
        cmd = "python3 " + run_file
        fb_list = [0, 107, 1684, 1912, 3437, 348, 3980, 414, 686, 698]
        call_list = cmd.split(" ")
        call_list.append("--dataset-str")
        call_list.append("")
        for i in range(len(fb_list)):
            fb_file = "facebook"+ str(fb_list[i])
            call_list[-1] = fb_file
            call(call_list)
            with open(log_file, "r") as f:
                result = f.readlines()[-1]
                f.close() 
            output = "Exp {i}: ".format(i = i, lr = 0.05) + result
            with open(result_file,"a") as f:
                f.write(output)
                f.close()



if __name__ == "__main__":
    args = parser.parse_args()
    overlap = args.overlap
    multi_hyper = args.multi_hyper

    run_experiment(overlap,multi_hyper)





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
# count = 0
# # for i in range(lr_index):
# for j in range(dp_index):
#     count += 1
#     overlap_cmd = "python3 nonoverlapping-community-detection.py --lr {lr} --dropout {dropout}".format(lr = 0.05, dropout = dropouts[j])
#     overlap_call_list = overlap_cmd.split(" ")
#     call(overlap_call_list)
#     with open("nonoverlapping.log","r") as f:
#         result = f.readlines()[-1]
#         f.close()

#     output = "Exp {i}: lr = {lr}, dropout = {dropout} ".format(i = count, lr = 0.05, dropout = dropouts[j]) + result
#     with open("nonoverlapping_exp_results.log","a") as f:
#         f.write(output)
#         f.close()


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


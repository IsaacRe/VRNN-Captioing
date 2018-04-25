import numpy as np
def parse_log(log_file):
    with open(log_file) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    epoch_counter = 0
                
    # generate average train loss and average train accuracy across all files.
    train_avg_losses = []
    train_step_losses = []
    train_step_perplexity = []
    eval_avg_losses = []

    for line in content:
        words = line.split()
        if words[0] == "Epoch" and words[2] == "Step":
            train_step_losses.append(float(words[5].replace(",", "")))
            train_step_perplexity.append(float(words[-1].replace(",", "")))
        if words[0] == "Avg" and words[1] == "Train":
            train_avg_losses.append(float(words[3].replace(",", "")))
        if words[0] == "Avg" and words[1] == "Eval":
            eval_avg_losses.append(float(words[3].replace(",", "")))
            
    print "%d training, %d eval"%(len(train_avg_losses), len(eval_avg_losses))
    return train_avg_losses, train_step_losses, train_step_perplexity, eval_avg_losses

import matplotlib.pyplot as plt
def plot(curves, names, graph_title = ''):
    print "There are %d elements"%(len(curves[0]))
    if len(curves) is not len(names):
        print "# curves does not match # names: %d vs %d"%(len(curves), len(names))
        return
    # deal with missing data.
    #num_epoch = 0
    #for i in len(curves):
    #    num_epoch = max(num_epoch, len(curves[i]))
    #    for j in range(len(curves[i]), num_epoch):
    #        curves[i].append(0)
    for i in range(len(curves)):
        plt.plot(curves[i], label=names[i])
        ind = np.argmin(curves[i])
        print "best %s is %.4f at step %d"%(names[i], curves[i][ind]*100, ind)
    plt.legend(loc='lower right')
    plt.title(graph_title)
    plt.show()
    
def plot_losses(filename):
    train_avg_losses, train_step_losses, train_step_perplexity, eval_avg_losses = parse_log(filename)
    
    if len(train_avg_losses) >0 and len(eval_avg_losses) > 0:
        plot([train_avg_losses, eval_avg_losses], ['train average loss', 'eval average loss'])

    if len(train_step_losses) > 0:
        plot([train_step_losses], ['train step loss'])

    if len(train_step_perplexity)>0:
        plot([train_step_perplexity[1:]], ['train step perplexity (except first one)'])

def plot_overlay(log_files):
    train_losses = []
    names = []
    eval_losses = []
    for log_file in log_files:
        log_name = log_file[0]
        names.append(log_name)
        file_path = log_file[1]
        train_avg_losses, train_step_losses, train_step_perplexity, eval_avg_losses = parse_log(file_path)
        train_losses.append(train_step_losses)
        eval_losses.append(train_step_perplexity)
    plot(train_losses, names, graph_title='train loss overlay')
    plot(eval_losses, names, graph_title = 'train perplexity overlay')
from imports import *

def train_mod(train_loader, model, optimizer, error,
              save_dir, num_epochs=1, print_every=100):
    """
    method for training model.

    PARAMS
    -----------------------------
    epochs: number of epochs to trains for
    print_every: how often to print accuracy and loss
    save_dir: directory to save model checkpoints
    """
    # train model
    loss_list = []
    count = 0
    iteration_list = []
    accuracy_list = []
    for epoch in range(num_epochs):
            for batch in train_loader:
                x_batch = batch[:1]
                y_batch = batch[-1]
                optimizer.zero_grad()
                outputs = model(*x_batch)
                loss = error(outputs, y_batch)
                loss.backward()
                optimizer.step()
                count += 1
                loss_list.append(loss)
                iteration_list.append(count)
                accuracy = int(((outputs>.5).float().squeeze() == y_batch).sum())/5
                accuracy_list.append(accuracy)
                if count % 100 == 0:
                    print('iteration: {} loss: {} accuracy: {}'.format(count, loss, sum(accuracy_list)/(count)))
                if count % 300 == 0:
                    torch.save(model.state_dict(), save_dir)
    return iteration_list, loss_list, accuracy_list



def plots(output_dir):
    """
    plots accuracy and loss over iterations

    PARAMS
    ------------------------------
    output_dir: output directory where figures should be saved
    """
    plt.plot(iteration_list, loss_list)
    plt.xlabel("iterations")
    plt.ylabel("loss")
    plt.save_fig(output_dir + "loss_iterations.png")
    plt.plot(iteration_list, accuracy_list)
    plt.xlabel("iterations")
    plt.ylabel("accuracy")
    plot.save_fig(output_dir + "accurcy_iterations.png")



import torch
import torch.nn as nn

criterion = nn.CrossEntropyLoss()
criterion_sum = nn.CrossEntropyLoss(reduction='sum')        # for validation and test


# function for training the model
def train(model, train_loader, val_loader,  optimizer, args):
    model.train()
    train_loss_list = []
    val_loss_list = []
    best_val_acc = 0
    for ep in range(1, args.epochs + 1):
        correct = 0
        train_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output, _ = model(data)
            loss = criterion(output, target)
            train_loss += loss.item()
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
            loss.backward()
            optimizer.step()

        train_loss /= len(train_loader)
        train_loss_list.append(train_loss)
        val_loss, val_acc = validate(model, val_loader, args)
        val_loss_list.append(val_loss)
        if ep % 10 == 0:
            print('\nTrain epoch: {}, Average train loss: {:.4f}, Train Accuracy: {}/{} ({:.0f}%), Average validation loss: {:.4f}, Validation Accuracy:({:.0f}%)\n'.format(ep,
                train_loss, correct, len(train_loader.dataset),
                100. * correct / len(train_loader.dataset), val_loss, val_acc))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # saving as a checkpoint to enable further training
            torch.save({
                'epoch': ep,
                'lr': args.lr,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_acc,
                'bs': args.batch_size,
                'hidden_units': args.hidden_units_l1
            }, args.save_path)
            print('\nModel saved at epoch: {}, with validation accuracy: {:.0f}\n'.format(ep, 100*val_acc))
    return train_loss_list, val_loss_list


# function to run validation and tests
def validate(model, data_loader, args):
    model.eval()
    val_loss = 0
    correct = 0
    for data, target in data_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        output, _ = model(data)
        val_loss += criterion_sum(output, target).item()  # sum batch-loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

    val_loss /= len(data_loader.dataset)
    return val_loss, 100. * correct / len(data_loader.dataset)


# function to load a save model -> can be used for further training  or evaluation
def load_model(model, optimizer, file_loc):
    checkpoint = torch.load(file_loc)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    val_accuracy = checkpoint['val_accuracy']
    print('Loaded model had best validation accuracy: ', val_accuracy.item())
    return model, optimizer

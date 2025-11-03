import torch
import numpy as np
from tqdm.auto import tqdm
from PatchEmbeds import PatchEmbeddings,patcher

def train_phase(model,train_loader,optimizer,loss_fn,schedular):
    model.train()
    batch_loss,batch_acc = [],[]
    global_steps = 0
    for images,labels in train_loader:
        image_patches = []
        for image in images:
            image_patches.append(patcher(image))
        patch_embeddings = PatchEmbeddings(patches=image_patches,number_of_patches=16,out_features=768)
        yhat = model(patch_embeddings)
        loss = loss_fn(yhat,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        schedular.step()
        global_steps += 1
        batch_loss.append(loss.item())
        probs = torch.argmax(torch.softmax(yhat,dim=1),dim=1)
        accuracy = 100 * (probs == labels).float().mean()
        batch_acc.append(accuracy.item())
    return batch_loss,batch_acc,model,global_steps



def train_and_test(model,train_loader,optimizer,loss_fn,schedular,test_loader=None,epochs=100,device=None):
    results = {
        "average_train_batch_loss": [],
        "average_train_batch_acc": [],
        "total_train_loss": [],
        "total_tain_acc" : []
    }
    global_steps = 0
    for epoch in tqdm(range(epochs)):
        batch_loss,batch_acc,trained_model,global_steps = train_phase(model,train_loader=train_loader,optimizer=optimizer,loss_fn=loss_fn,schedular=schedular)
        # call and implement test phase and inlcude its keys in results
        global_steps += global_steps
        results['average_train_batch_loss'].append(np.mean(batch_loss))
        results['average_train_batch_acc'].append(np.mean(batch_acc))
        results['total_train_loss'].extend(batch_loss)
        results['total_tain_acc'].extend(batch_acc)

        print(
            f"Epoch: {epoch + 1} | "
            f"Train Accuracy: {results['average_train_batch_loss']:.1f}% | "
            f"Train Loss: {results['average_train_batch_acc']:.4f} | "
            f"Global steps: {global_steps} |"
        )

    return results,trained_model
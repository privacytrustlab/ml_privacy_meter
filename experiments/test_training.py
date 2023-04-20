from models import get_model
from dataset import get_dataset, get_dataset_subset, get_dataloader
from core import prepare_datasets_for_online_attack
import numpy as np
import torch
import time
from train import train, inference


if __name__ == "__main__":
    np.random.seed(1022321)
    batch_size = 256
    model_name = "wrn28-2"
    dataset = get_dataset("cifar10", "../data")
    p_ratio = 0.5
    dataset_size = 50000
    # data_split_info, keep_matrix = prepare_datasets_for_online_attack(
    #     dataset_size,
    #     num_models=1,
    #     keep_ratio=0.5,
    #     configs=None,
    #     model_metadata_dict=None,
    # )
    # data, targets = get_dataset_subset(
    #     dataset, np.arange(dataset_size)
    # )  # only the train dataset we want to attack
    train_index = np.random.choice(np.arange(dataset_size), int(dataset_size * p_ratio), replace=False)
    
        
    train_loader = get_dataloader(
        torch.utils.data.Subset(dataset, train_index),
        batch_size=batch_size,
        shuffle=True,
       
    )
    test_loader = get_dataloader(
        torch.utils.data.Subset(dataset, [i for i in range(50000, 60000)]),
        batch_size=10000,
       
    )

    baseline_time = time.time()
    # Train the target model based on the configurations.
    config = {"epochs": 100, "learning_rate": 0.1, "device": "cuda:1", "optimizer": "SGD", "weight_decay":0.0005, "momentum": 0.9}
    model = train(get_model(model_name), train_loader, config, test_loader)
    
    # model = get_model(model_name)
    # ema_handler = EMAHandler(model, momentum=0.0002)
    # ema_model = ema_handler.ema_model
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    # trainer = create_supervised_trainer(model=model, optimizer=optimizer, loss_fn=torch.nn.CrossEntropyLoss())
    # ema_handler.attach(trainer, name="ema_momentum", event=Events.ITERATION_COMPLETED(every=1))

    # to_load = {"model": model, "ema_model", ema_model, "trainer", trainer}

    # Train the model as usual
    # for epoch in range(2):
    #     trainer.run(train_loader)
    
    
    # Test performance on the training dataset and test dataset
    test_loss, test_acc = inference(model, test_loader, "cuda:1")
    train_loss, train_acc = inference(model, train_loader, "cuda:1")
    print(f"Train accuracy {train_acc}, Train Loss {train_loss}")
    print(f"Test accuracy {test_acc}, Test Loss {test_loss}")
    print(f"Baseline time {time.time() - baseline_time}")
    
    # Test performance on the training dataset and test dataset
    test_loss, test_acc = inference(model, test_loader, "cuda:1")
    train_loss, train_acc = inference(model, train_loader, "cuda:1")
    print(f"Train accuracy {train_acc}, Train Loss {train_loss}")
    print(f"Test accuracy {test_acc}, Test Loss {test_loss}")
    print(f"Baseline time {time.time() - baseline_time}")

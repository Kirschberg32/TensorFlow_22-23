import tqdm
import tensorflow as tf

def training_loop(model, train_ds, epochs_start, epochs_end, starting_prompt, output_length, top_k, train_summary_writer, val_summary_writer,save_path):
    for epoch in range(epochs_start,epochs_end):
        print(f"Epoch {epoch}:")
        
        # Training:
        for data in tqdm.tqdm(train_ds, position=0, leave=True):
            metrics = model.train_step(data)
            
            # logging the validation metrics to the log file which is used by tensorboard
            with train_summary_writer.as_default():
                for metric in model.metrics:
                    tf.summary.scalar(f"{metric.name}", metric.result(), step=epoch)

        # print the metrics
        print([f"{key}: {value.numpy()}" for (key, value) in metrics.items()])

        # reset all metrics (requires a reset_metrics method in the model)
        model.reset_metrics()    
        
        # Validation: text generation
        text = model.generate_text(starting_prompt,output_length, top_k)
        print("Generated Text: ", text)

        tf.summary.text(f"Epoch {epoch}",text, step=epoch)

        # reset all metrics
        model.reset_metrics()
        print("\n")

        # save model
        if epoch%10 is 0:
            tf.keras.models.save_model(model,save_path)
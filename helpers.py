import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report


def plot_misclassification(model, X_test, y_test, class_names, num_images=25):
    """
    afiseaza cateva imagini clasificate gresit
    """
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_flat = y_test.flatten()

    misclassified_indices = np.where(y_pred_classes != y_test_flat)[0]

    plt.figure(figsize=(10, 10))

    for i in range(min(num_images, len(misclassified_indices))):
        idx = misclassified_indices[i]
        plt.subplot(4, 4, i + 1)

        plt.imshow(X_test[idx])
        plt.axis("off")

        true_label = class_names[y_test_flat[idx]]
        pred_label = class_names[y_pred_classes[idx]]
        plt.title(f"Adevarat: {true_label}\nPrezis: {pred_label}", fontsize=12)

    plt.tight_layout()
    plt.show()


def plot_learning_curve(history):
    """
    afiseaza curbele de invatare pentru acuratete si pierdere
    """
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, "b-", label="Acuratete antrenare")
    plt.plot(epochs, val_acc, "r-", label="Acuratete validare")
    plt.title("Acuratete in functie de epoca")
    plt.xlabel("Epoca")
    plt.ylabel("Acuratete")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, "b-", label="Pierdere antrenare")
    plt.plot(epochs, val_loss, "r-", label="Pierdere validare")
    plt.title("Pierdere in functie de epoca")
    plt.xlabel("Epoca")
    plt.ylabel("Pierdere")
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(model, X_test, y_test, class_names):
    """
    genereaza predictii pe un model dat.
    afiseaza matricea de confuzie si raportul de clasificare
    """

    # predictii
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # flatten
    y_test_flat = y_test.flatten()

    # raport de clasificare
    print(classification_report(y_test_flat, y_pred_classes, target_names=class_names))

    # matrice de confuzie
    cm = confusion_matrix(y_test_flat, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Prezis")
    plt.ylabel("Adevarat")
    plt.title("Matrice de confuzie")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

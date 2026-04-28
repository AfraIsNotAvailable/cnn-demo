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


def plot_confidence_distribution(model, X_test, y_test):
    """arata nivelul de confidenta cu care prezice modelul fiecare clasa"""
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = y_test.flatten()

    # extrage probabilitatile pentru clasele prezise
    max_probs = np.max(y_pred, axis=1)

    # separa in doua grupuri: corect prezise si gresit prezise
    correct_confidences = max_probs[y_pred_classes == y_true]
    incorrect_confidences = max_probs[y_pred_classes != y_true]

    # afiseaza distributia confidentelor
    plt.figure(figsize=(8, 5))
    sns.kdeplot(
        correct_confidences,
        label="Corect prezise",
        fill=True,
        color="green",
        alpha=0.5,
    )
    sns.kdeplot(
        incorrect_confidences, label="Gresit prezise", fill=True, color="red", alpha=0.5
    )

    plt.title("Confidenta")
    plt.xlabel("Nivel de confidenta")
    plt.ylabel("Densitate")
    plt.legend()
    plt.xlim(0, 1)
    plt.show()


def plot_overfitting(train_acc, val_acc, epochs=None):
    """afiseaza curbele de invatare pentru acuratete si pierdere, evidentiind overfitting-ul"""
    best_epoch = int(np.argmax(val_acc))
    best_val = val_acc[best_epoch]
    best_train = train_acc[best_epoch]
    delta = best_train - best_val

    x_dim = best_epoch + 1  # adaugam un mic offset pentru a face loc textului

    plt.figure(figsize=(12, 6))
    plt.plot(train_acc, label="Training Accuracy")
    plt.plot(val_acc, label="Validation Accuracy")
    plt.axvline(
        x=best_epoch,
        color="purple",
        linestyle="--",
        alpha=0.7,
        label=f"Best Epoch: {best_epoch}",
    )

    plt.plot([best_epoch, x_dim], [best_train, best_train], "k-", lw=1, alpha=0.7)
    plt.plot([best_epoch, x_dim], [best_val, best_val], "k-", lw=1, alpha=0.7)

    plt.annotate(
        "",
        xy=(x_dim, best_train),
        xytext=(x_dim, best_val),
        arrowprops=dict(arrowstyle="<->", color="black", lw=1),
    )

    mid = (best_train + best_val) / 2
    plt.text(
        x_dim + 0.5,
        mid,
        f"Δ = {delta:.2f}",
        va="center",
        ha="left",
        fontsize=12,
        fontfamily="monospace",
    )

    # adaugam axvlien pentru fiecare rulare dupa numarul de epoci per rulare ca sa vedem cat a crescut
    if epochs is not None:
        for e in epochs:
            plt.axvline(x=e, color="gray", linestyle=":", alpha=0.5)

    plt.ylim(0.35, 1)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(
        f"Overfitting Visualization - Best Val Acc: {best_val:.4f} at Epoch {best_epoch}"
    )
    plt.legend(loc="lower right")
    plt.show()


def show_augmentations_bad(image, label, class_names):
    """varianta initiala (naiva): rotatii pe 90° si flip vertical produc imagini
    nerealiste (camioane cu rotile in sus). pastrata pentru comparatie didactica.
    4 rotatii (0°, 90°, 180°, 270°) x 3 stari flip (niciuna/H/V) x 4 translatii (U/D/L/R) = 48
    """

    def _translate(img, direction):
        out = np.zeros_like(img)
        if direction == "U":
            out[:-1, :] = img[1:, :]
        elif direction == "D":
            out[1:, :] = img[:-1, :]
        elif direction == "L":
            out[:, :-1] = img[:, 1:]
        elif direction == "R":
            out[:, 1:] = img[:, :-1]
        return out

    rot_labels = ["0°", "90°", "180°", "270°"]
    flip_fns = [
        (None, ""),
        (np.fliplr, "+FlipH"),
        (np.flipud, "+FlipV"),
    ]
    trans_dirs = [("U", "↑"), ("D", "↓"), ("L", "←"), ("R", "→")]

    variants = []
    for k, rl in enumerate(rot_labels):
        rot_img = np.rot90(image, k)
        for flip_fn, fl in flip_fns:
            flipped = flip_fn(rot_img) if flip_fn is not None else rot_img
            for td, tl in trans_dirs:
                variants.append((_translate(flipped, td), f"{rl}{fl}{tl}"))

    label_idx = int(np.asarray(label).flatten()[0])
    display_img = np.clip(image, 0, 1)

    fig, axes = plt.subplots(7, 7, figsize=(14, 14))
    axes = axes.flatten()

    axes[0].imshow(display_img)
    axes[0].set_title(
        f"{class_names[label_idx]}\n(original)", fontsize=8, fontweight="bold"
    )
    axes[0].axis("off")

    for i, (var_img, title) in enumerate(variants):
        axes[i + 1].imshow(np.clip(var_img, 0, 1))
        axes[i + 1].set_title(title, fontsize=5)
        axes[i + 1].axis("off")

    plt.suptitle(
        "Augmentari naive: 4 rotatii × 3 flip × 4 translatii (48)", fontsize=10, y=1.01
    )
    plt.tight_layout()
    plt.show()


def show_augmentations(image, label, class_names):
    """afiseaza imaginea originala si cele 40 de variatii ale ei:
    5 rotatii (-20°, -10°, 0°, 10°, 20°) x 2 stari flip (niciuna/H) x 4 translatii (U/D/L/R) = 40
    """
    from scipy.ndimage import rotate as nd_rotate

    def _translate(img, direction):
        out = np.zeros_like(img)
        if direction == "U":
            out[:-1, :] = img[1:, :]
        elif direction == "D":
            out[1:, :] = img[:-1, :]
        elif direction == "L":
            out[:, :-1] = img[:, 1:]
        elif direction == "R":
            out[:, 1:] = img[:, :-1]
        return out

    angles = [-20, -10, 0, 10, 20]
    flip_fns = [(None, ""), (np.fliplr, "+FlipH")]
    trans_dirs = [("U", "↑"), ("D", "↓"), ("L", "←"), ("R", "→")]

    variants = []
    for angle in angles:
        if angle == 0:
            rot_img = image
        else:
            rot_img = nd_rotate(
                image, angle, reshape=False, mode="constant", cval=0.0, order=1
            )
        rl = f"{angle:+d}°" if angle else "0°"
        for flip_fn, fl in flip_fns:
            flipped = flip_fn(rot_img) if flip_fn is not None else rot_img
            for td, tl in trans_dirs:
                variants.append((_translate(flipped, td), f"{rl}{fl}{tl}"))

    label_idx = int(np.asarray(label).flatten()[0])
    display_img = np.clip(image, 0, 1)

    fig, axes = plt.subplots(6, 7, figsize=(14, 12))
    axes = axes.flatten()

    axes[0].imshow(display_img)
    axes[0].set_title(
        f"{class_names[label_idx]}\n(original)", fontsize=8, fontweight="bold"
    )
    axes[0].axis("off")

    for i, (var_img, title) in enumerate(variants):
        axes[i + 1].imshow(np.clip(var_img, 0, 1))
        axes[i + 1].set_title(title, fontsize=12)
        axes[i + 1].axis("off")

    for j in range(len(variants) + 1, len(axes)):
        axes[j].axis("off")

    plt.suptitle("Augmentari: 5 rotatii × 2 flip × 4 translatii", fontsize=10, y=1.01)
    plt.tight_layout()
    plt.show()


def compare_models(models_dict, X_test, y_test):
    """compara performanta mai multor modele pe setul de test.
    afiseaza bar chart cu acuratete si pierdere, plus tabel cu valorile.

    models_dict: dict {nume: model_keras} (ex: {"Naiv": naive_model, "Dropout": d_model})
    """
    names = list(models_dict.keys())
    accuracies = []
    losses = []

    for name, model in models_dict.items():
        # accepta si History (returnat de .fit()) - extrage modelul real
        if hasattr(model, "model"):
            model = model.model
        loss, acc = model.evaluate(X_test, y_test, verbose=0)
        accuracies.append(acc)
        losses.append(loss)
        print(f"{name:25s}  acc={acc:.4f}  loss={loss:.4f}")

    colors = sns.color_palette("Set2", len(names))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    bars_acc = axes[0].bar(names, accuracies, color=colors)
    axes[0].set_title("Acuratete pe setul de test")
    axes[0].set_ylabel("Acuratete")
    axes[0].set_ylim(0, 1)
    for bar, val in zip(bars_acc, accuracies):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.01,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    bars_loss = axes[1].bar(names, losses, color=colors)
    axes[1].set_title("Pierdere pe setul de test")
    axes[1].set_ylabel("Pierdere")
    for bar, val in zip(bars_loss, losses):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            val + max(losses) * 0.01,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    for ax in axes:
        ax.tick_params(axis="x", rotation=15)

    plt.tight_layout()
    plt.show()


def show_dataset(train_images, train_labels, class_names, num_images=25):
    """afiseaza cateva imagini din dataset cu etichetele lor"""
    plt.figure(figsize=(10, 10))
    for i in range(num_images):
        plt.subplot(5, 5, i + 1)
        plt.imshow(train_images[i])
        plt.axis("off")
        label = class_names[train_labels[i][0]]
        plt.title(label, fontsize=12)
    plt.tight_layout()
    plt.show()

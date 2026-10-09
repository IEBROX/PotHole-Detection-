import sys
import time
import csv
from pathlib import Path

import cv2
import yaml

from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QFileDialog,
    QProgressBar,
    QFrame,
    QSizePolicy,
    QComboBox,
    QMessageBox,
    QDialog,
    QScrollArea,
    QTabWidget,
    QGridLayout,
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap, QImage

from ultralytics import YOLO


# ============================================================
# ZOOMABLE IMAGE LABEL
# ============================================================

class ZoomableLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.original_pixmap = None
        self.zoom_factor = 1.0

        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(300, 250)
        self.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding
        )

    def set_image(self, pixmap):
        self.original_pixmap = pixmap
        self.zoom_factor = 1.0
        self.update_display()

    def update_display(self):
        if self.original_pixmap is None:
            return

        if self.width() <= 10 or self.height() <= 10:
            return

        scaled = self.original_pixmap.scaled(
            int(self.width() * self.zoom_factor),
            int(self.height() * self.zoom_factor),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        self.setPixmap(scaled)

    def zoom_in(self):
        self.zoom_factor = min(
            self.zoom_factor + 0.1,
            3.0
        )
        self.update_display()

    def zoom_out(self):
        self.zoom_factor = max(
            self.zoom_factor - 0.1,
            0.5
        )
        self.update_display()

    def reset_zoom(self):
        self.zoom_factor = 1.0
        self.update_display()

    def fit_image(self):
        self.zoom_factor = 1.0
        self.update_display()

    def wheelEvent(self, event):
        if event.angleDelta().y() > 0:
            self.zoom_in()
        else:
            self.zoom_out()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_display()


# ============================================================
# MODEL DETAILS DIALOG
# ============================================================

class ModelDetailsDialog(QDialog):

    def __init__(self, model_path, details_dir, model=None, parent=None):
        super().__init__(parent)

        self.model_path = Path(model_path)
        self.details_dir = Path(details_dir)
        self.model = model

        self.setWindowTitle("Model Details")
        self.resize(1000, 720)

        self.args = {}
        self.results = []
        self.final_metrics = {}
        self.best_metrics = {}
        self.best_epoch = "-"

        self.load_training_details()

        self.setup_ui()

    # --------------------------------------------------------
    # LOAD YAML + CSV
    # --------------------------------------------------------

    def load_training_details(self):

        # -------------------------
        # args.yaml
        # -------------------------

        args_file = self.details_dir / "args.yaml"

        if args_file.exists():
            try:
                with open(
                    args_file,
                    "r",
                    encoding="utf-8"
                ) as f:
                    self.args = yaml.safe_load(f) or {}

            except Exception:
                self.args = {}

        # -------------------------
        # results.csv
        # -------------------------

        results_file = self.details_dir / "results.csv"

        if results_file.exists():

            try:
                with open(
                    results_file,
                    "r",
                    encoding="utf-8-sig"
                ) as f:

                    reader = csv.DictReader(f)

                    for row in reader:
                        self.results.append(row)

            except Exception:
                self.results = []

        # -------------------------
        # Final metrics
        # -------------------------

        if self.results:
            self.final_metrics = self.results[-1]

            # Find best epoch based on mAP50-95
            map_key = self.find_column(
                [
                    "metrics/mAP50-95(B)",
                    "metrics/mAP50-95",
                ]
            )

            if map_key:

                best_row = None
                best_value = -1

                for row in self.results:

                    try:
                        value = float(
                            row.get(map_key, 0)
                        )

                        if value > best_value:
                            best_value = value
                            best_row = row

                    except (ValueError, TypeError):
                        pass

                if best_row:

                    self.best_metrics = best_row

                    epoch_value = best_row.get(
                        "epoch",
                        "-"
                    )

                    self.best_epoch = epoch_value

    # --------------------------------------------------------
    # FIND CSV COLUMN
    # --------------------------------------------------------

    def find_column(self, possible_names):

        for name in possible_names:

            if name in self.final_metrics:
                return name

        return None

    # --------------------------------------------------------
    # GET VALUE
    # --------------------------------------------------------

    def get_metric(self, possible_names):

        column = self.find_column(possible_names)

        if not column:
            return "-"

        value = self.final_metrics.get(
            column,
            "-"
        )

        try:
            return float(value)

        except (ValueError, TypeError):
            return value

    # --------------------------------------------------------
    # FORMAT METRIC
    # --------------------------------------------------------

    def format_metric(self, value):

        if value == "-":
            return "-"

        try:
            return f"{float(value):.4f}"

        except (ValueError, TypeError):
            return str(value)

    # --------------------------------------------------------
    # CREATE INFO CARD
    # --------------------------------------------------------

    def create_info_card(
        self,
        title,
        value,
        description=""
    ):

        card = QFrame()
        card.setObjectName("detailsCard")

        layout = QVBoxLayout(card)
        layout.setContentsMargins(
            16, 14, 16, 14
        )
        layout.setSpacing(5)

        title_label = QLabel(title.upper())
        title_label.setObjectName(
            "detailsCaption"
        )

        value_label = QLabel(str(value))
        value_label.setObjectName(
            "detailsValue"
        )
        value_label.setWordWrap(True)

        layout.addWidget(title_label)
        layout.addWidget(value_label)

        if description:

            desc_label = QLabel(description)
            desc_label.setObjectName(
                "detailsDescription"
            )
            desc_label.setWordWrap(True)

            layout.addWidget(desc_label)

        return card

    # --------------------------------------------------------
    # IMAGE VIEWER
    # --------------------------------------------------------

    def create_image_page(
        self,
        image_path,
        title,
        description=""
    ):

        page = QWidget()

        layout = QVBoxLayout(page)
        layout.setContentsMargins(
            16, 16, 16, 16
        )

        if description:

            desc = QLabel(description)
            desc.setObjectName(
                "detailsPageDescription"
            )
            desc.setWordWrap(True)

            layout.addWidget(desc)

        image_label = QLabel()
        image_label.setAlignment(
            Qt.AlignCenter
        )
        image_label.setMinimumSize(
            600,
            450
        )

        image_label.setStyleSheet("""
            QLabel {
                background-color: #07080C;
                border: 1px solid #272A38;
                border-radius: 10px;
            }
        """)

        image_path = Path(image_path)

        if image_path.exists():

            pixmap = QPixmap(
                str(image_path)
            )

            if not pixmap.isNull():

                image_label.setPixmap(
                    pixmap.scaled(
                        850,
                        520,
                        Qt.KeepAspectRatio,
                        Qt.SmoothTransformation
                    )
                )

        else:

            image_label.setText(
                "Image not available"
            )

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(image_label)

        layout.addWidget(scroll)

        return page

    # --------------------------------------------------------
    # SETUP UI
    # --------------------------------------------------------

    def setup_ui(self):

        main_layout = QVBoxLayout(self)

        main_layout.setContentsMargins(
            18, 18, 18, 18
        )

        main_layout.setSpacing(12)

        # ====================================================
        # HEADER
        # ====================================================

        header = QHBoxLayout()

        title = QLabel(
            "MODEL DETAILS"
        )
        title.setObjectName(
            "detailsTitle"
        )

        subtitle = QLabel(
            "Training configuration and evaluation results"
        )
        subtitle.setObjectName(
            "detailsSubtitle"
        )

        header_text = QVBoxLayout()
        header_text.setSpacing(2)

        header_text.addWidget(title)
        header_text.addWidget(subtitle)

        header.addLayout(header_text)
        header.addStretch()

        main_layout.addLayout(header)

        # ====================================================
        # TABS
        # ====================================================

        tabs = QTabWidget()
        tabs.setObjectName("detailsTabs")

        # ====================================================
        # OVERVIEW TAB
        # ====================================================

        overview = QWidget()

        overview_layout = QVBoxLayout(
            overview
        )

        overview_scroll = QScrollArea()
        overview_scroll.setWidgetResizable(True)

        overview_content = QWidget()

        content_layout = QVBoxLayout(
            overview_content
        )

        # ----------------------------------------------------
        # MODEL INFORMATION
        # ----------------------------------------------------

        model_section = QLabel(
            "MODEL INFORMATION"
        )
        model_section.setObjectName(
            "detailsSection"
        )

        content_layout.addWidget(
            model_section
        )

        model_grid = QGridLayout()
        model_grid.setSpacing(10)

        # Model name
        model_name = (
            self.model_path.stem
            if self.model_path
            else "Unknown"
        )

        model_grid.addWidget(
            self.create_info_card(
                "Model",
                model_name
            ),
            0,
            0
        )

        # Architecture
        architecture = self.args.get(
            "model",
            "YOLO"
        )

        if architecture:
            architecture = Path(
                str(architecture)
            ).stem

        model_grid.addWidget(
            self.create_info_card(
                "Architecture",
                architecture
            ),
            0,
            1
        )

        # Task
        model_grid.addWidget(
            self.create_info_card(
                "Task",
                "Object Detection"
            ),
            0,
            2
        )

        # Classes
        class_names = "Unknown"

        if self.model is not None:

            try:

                names = self.model.names

                if isinstance(names, dict):
                    class_names = ", ".join(
                        str(v)
                        for v in names.values()
                    )

                elif isinstance(names, list):
                    class_names = ", ".join(
                        str(v)
                        for v in names
                    )

            except Exception:
                pass

        model_grid.addWidget(
            self.create_info_card(
                "Classes",
                class_names
            ),
            1,
            0
        )

        # Model file
        model_grid.addWidget(
            self.create_info_card(
                "Model File",
                self.model_path.name
            ),
            1,
            1
        )

        # Platform
        model_grid.addWidget(
            self.create_info_card(
                "Training Platform",
                "Google Colab"
            ),
            1,
            2
        )

        content_layout.addLayout(
            model_grid
        )

        # ----------------------------------------------------
        # TRAINING CONFIGURATION
        # ----------------------------------------------------

        training_section = QLabel(
            "TRAINING CONFIGURATION"
        )
        training_section.setObjectName(
            "detailsSection"
        )

        content_layout.addSpacing(14)
        content_layout.addWidget(
            training_section
        )

        training_grid = QGridLayout()
        training_grid.setSpacing(10)

        epochs = self.args.get(
            "epochs",
            "-"
        )

        batch = self.args.get(
            "batch",
            "-"
        )

        imgsz = self.args.get(
            "imgsz",
            "-"
        )

        optimizer = self.args.get(
            "optimizer",
            "-"
        )

        lr0 = self.args.get(
            "lr0",
            "-"
        )

        device = self.args.get(
            "device",
            "-"
        )

        dataset = self.args.get(
            "data",
            "-"
        )

        training_values = [
            ("Epochs", epochs),
            ("Batch Size", batch),
            ("Image Size", imgsz),
            ("Optimizer", optimizer),
            ("Learning Rate", lr0),
            ("Device", device),
            ("Dataset", dataset),
        ]

        for index, (name, value) in enumerate(
            training_values
        ):

            row = index // 3
            col = index % 3

            training_grid.addWidget(
                self.create_info_card(
                    name,
                    value
                ),
                row,
                col
            )

        content_layout.addLayout(
            training_grid
        )

        # ----------------------------------------------------
        # PERFORMANCE
        # ----------------------------------------------------

        performance_section = QLabel(
            "PERFORMANCE"
        )
        performance_section.setObjectName(
            "detailsSection"
        )

        content_layout.addSpacing(14)

        content_layout.addWidget(
            performance_section
        )

        performance_grid = QGridLayout()
        performance_grid.setSpacing(10)

        precision = self.get_metric(
            [
                "metrics/precision(B)",
                "metrics/precision",
            ]
        )

        recall = self.get_metric(
            [
                "metrics/recall(B)",
                "metrics/recall",
            ]
        )

        map50 = self.get_metric(
            [
                "metrics/mAP50(B)",
                "metrics/mAP50",
            ]
        )

        map95 = self.get_metric(
            [
                "metrics/mAP50-95(B)",
                "metrics/mAP50-95",
            ]
        )

        performance_values = [
            (
                "Precision",
                self.format_metric(
                    precision
                )
            ),
            (
                "Recall",
                self.format_metric(
                    recall
                )
            ),
            (
                "mAP50",
                self.format_metric(
                    map50
                )
            ),
            (
                "mAP50-95",
                self.format_metric(
                    map95
                )
            ),
            (
                "Best Epoch",
                self.best_epoch
            ),
        ]

        for index, (name, value) in enumerate(
            performance_values
        ):

            performance_grid.addWidget(
                self.create_info_card(
                    name,
                    value
                ),
                index // 3,
                index % 3
            )

        content_layout.addLayout(
            performance_grid
        )

        # ----------------------------------------------------
        # FINAL LOSSES
        # ----------------------------------------------------

        loss_section = QLabel(
            "FINAL TRAINING LOSSES"
        )
        loss_section.setObjectName(
            "detailsSection"
        )

        content_layout.addSpacing(14)
        content_layout.addWidget(
            loss_section
        )

        loss_grid = QGridLayout()
        loss_grid.setSpacing(10)

        loss_values = [
            (
                "Train Box Loss",
                self.get_metric(
                    ["train/box_loss"]
                )
            ),
            (
                "Train Classification Loss",
                self.get_metric(
                    ["train/cls_loss"]
                )
            ),
            (
                "Train DFL Loss",
                self.get_metric(
                    ["train/dfl_loss"]
                )
            ),
            (
                "Validation Box Loss",
                self.get_metric(
                    ["val/box_loss"]
                )
            ),
            (
                "Validation Classification Loss",
                self.get_metric(
                    ["val/cls_loss"]
                )
            ),
            (
                "Validation DFL Loss",
                self.get_metric(
                    ["val/dfl_loss"]
                )
            ),
        ]

        for index, (name, value) in enumerate(
            loss_values
        ):

            loss_grid.addWidget(
                self.create_info_card(
                    name,
                    self.format_metric(
                        value
                    )
                ),
                index // 3,
                index % 3
            )

        content_layout.addLayout(
            loss_grid
        )

        content_layout.addStretch()

        overview_scroll.setWidget(
            overview_content
        )

        overview_layout.addWidget(
            overview_scroll
        )

        tabs.addTab(
            overview,
            "Overview"
        )

        # ====================================================
        # TRAINING RESULTS
        # ====================================================

        results_png = (
            self.details_dir /
            "results.png"
        )

        tabs.addTab(
            self.create_image_page(
                results_png,
                "Training Results",
                "Training and validation metrics recorded during model training."
            ),
            "Training Results"
        )

        # ====================================================
        # CONFUSION MATRIX
        # ====================================================

        confusion_png = (
            self.details_dir /
            "confusion_matrix.png"
        )

        tabs.addTab(
            self.create_image_page(
                confusion_png,
                "Confusion Matrix",
                "Shows how the trained model classified the validation data."
            ),
            "Confusion Matrix"
        )

        # ====================================================
        # NORMALIZED CONFUSION MATRIX
        # ====================================================

        normalized_png = (
            self.details_dir /
            "confusion_matrix_normalized.png"
        )

        tabs.addTab(
            self.create_image_page(
                normalized_png,
                "Normalized Confusion Matrix",
                "Normalized version of the confusion matrix."
            ),
            "Normalized Matrix"
        )

        # ====================================================
        # PRECISION / RECALL
        # ====================================================

        curves_page = QWidget()

        curves_layout = QVBoxLayout(
            curves_page
        )

        curves_scroll = QScrollArea()
        curves_scroll.setWidgetResizable(True)

        curves_content = QWidget()

        curves_grid = QGridLayout(
            curves_content
        )

        curve_files = [
            (
                "BoxF1_curve.png",
                "F1 Curve"
            ),
            (
                "BoxP_curve.png",
                "Precision Curve"
            ),
            (
                "BoxR_curve.png",
                "Recall Curve"
            ),
            (
                "BoxPR_curve.png",
                "Precision-Recall Curve"
            ),
        ]

        for index, (filename, title_text) in enumerate(
            curve_files
        ):

            card = QFrame()
            card.setObjectName(
                "graphCard"
            )

            card_layout = QVBoxLayout(card)

            title_label = QLabel(
                title_text
            )
            title_label.setObjectName(
                "graphTitle"
            )

            image_label = QLabel()
            image_label.setAlignment(
                Qt.AlignCenter
            )
            image_label.setMinimumHeight(
                280
            )

            path = (
                self.details_dir /
                filename
            )

            if path.exists():

                pixmap = QPixmap(
                    str(path)
                )

                image_label.setPixmap(
                    pixmap.scaled(
                        420,
                        280,
                        Qt.KeepAspectRatio,
                        Qt.SmoothTransformation
                    )
                )

            else:

                image_label.setText(
                    "Graph not available"
                )

            card_layout.addWidget(
                title_label
            )

            card_layout.addWidget(
                image_label
            )

            curves_grid.addWidget(
                card,
                index // 2,
                index % 2
            )

        curves_scroll.setWidget(
            curves_content
        )

        curves_layout.addWidget(
            curves_scroll
        )

        tabs.addTab(
            curves_page,
            "Performance Curves"
        )

        # ====================================================
        # DATASET
        # ====================================================

        dataset_page = QWidget()

        dataset_layout = QVBoxLayout(
            dataset_page
        )

        dataset_scroll = QScrollArea()
        dataset_scroll.setWidgetResizable(
            True
        )

        dataset_content = QWidget()

        dataset_grid = QGridLayout(
            dataset_content
        )

        dataset_images = [
            (
                "labels.jpg",
                "Dataset Labels"
            ),
            (
                "train_batch0.jpg",
                "Training Batch"
            ),
            (
                "train_batch1.jpg",
                "Training Batch 2"
            ),
            (
                "val_batch0_labels.jpg",
                "Validation Labels"
            ),
            (
                "val_batch0_pred.jpg",
                "Validation Predictions"
            ),
        ]

        for index, (filename, title_text) in enumerate(
            dataset_images
        ):

            card = QFrame()
            card.setObjectName(
                "graphCard"
            )

            card_layout = QVBoxLayout(card)

            title_label = QLabel(
                title_text
            )
            title_label.setObjectName(
                "graphTitle"
            )

            image_label = QLabel()
            image_label.setAlignment(
                Qt.AlignCenter
            )
            image_label.setMinimumHeight(
                280
            )

            path = (
                self.details_dir /
                filename
            )

            if path.exists():

                pixmap = QPixmap(
                    str(path)
                )

                image_label.setPixmap(
                    pixmap.scaled(
                        420,
                        280,
                        Qt.KeepAspectRatio,
                        Qt.SmoothTransformation
                    )
                )

            else:

                image_label.setText(
                    "Image not available"
                )

            card_layout.addWidget(
                title_label
            )

            card_layout.addWidget(
                image_label
            )

            dataset_grid.addWidget(
                card,
                index // 2,
                index % 2
            )

        dataset_scroll.setWidget(
            dataset_content
        )

        dataset_layout.addWidget(
            dataset_scroll
        )

        tabs.addTab(
            dataset_page,
            "Dataset"
        )

        main_layout.addWidget(
            tabs
        )

        # ====================================================
        # CLOSE
        # ====================================================

        close_button = QPushButton(
            "CLOSE"
        )
        close_button.setObjectName(
            "detailsCloseButton"
        )
        close_button.clicked.connect(
            self.accept
        )

        main_layout.addWidget(
            close_button,
            alignment=Qt.AlignRight
        )

        # ====================================================
        # STYLE
        # ====================================================

        self.setStyleSheet("""

            QDialog {
                background-color: #0A0B10;
                color: #F5F5F7;
            }

            QLabel#detailsTitle {
                font-size: 22px;
                font-weight: 700;
                color: #F5F5F7;
            }

            QLabel#detailsSubtitle {
                font-size: 11px;
                color: #8B8FA3;
            }

            QLabel#detailsSection {
                font-size: 11px;
                font-weight: 700;
                color: #A1A5B5;
                letter-spacing: 1px;
                padding-top: 8px;
            }

            QFrame#detailsCard {
                background-color: #171923;
                border: 1px solid #252837;
                border-radius: 10px;
            }

            QFrame#detailsCard:hover {
                border: 1px solid #34384C;
            }

            QLabel#detailsCaption {
                font-size: 8px;
                font-weight: 700;
                color: #85899C;
                letter-spacing: 0.8px;
            }

            QLabel#detailsValue {
                font-size: 14px;
                font-weight: 600;
                color: #F5F5F7;
            }

            QLabel#detailsDescription {
                font-size: 9px;
                color: #777B8D;
            }

            QLabel#detailsPageDescription {
                font-size: 10px;
                color: #8B8FA3;
                padding-bottom: 6px;
            }

            QTabWidget::pane {
                border: 1px solid #272A38;
                border-radius: 10px;
                background-color: #11131A;
            }

            QTabBar::tab {
                background-color: #171923;
                color: #85899C;
                padding: 9px 14px;
                margin-right: 2px;
                border-top-left-radius: 7px;
                border-top-right-radius: 7px;
                font-size: 10px;
                font-weight: 600;
            }

            QTabBar::tab:selected {
                background-color: #211C31;
                color: #C4B5FD;
            }

            QTabBar::tab:hover {
                color: #FFFFFF;
            }

            QScrollArea {
                border: none;
                background-color: transparent;
            }

            QFrame#graphCard {
                background-color: #171923;
                border: 1px solid #252837;
                border-radius: 10px;
            }

            QLabel#graphTitle {
                font-size: 11px;
                font-weight: 700;
                color: #D9DBE5;
                padding: 4px;
            }

            QPushButton#detailsCloseButton {
                min-width: 90px;
                min-height: 32px;
                max-height: 34px;
                border-radius: 7px;
                background-color: #8B5CF6;
                color: #FFFFFF;
                border: none;
                font-size: 10px;
                font-weight: 700;
            }

            QPushButton#detailsCloseButton:hover {
                background-color: #7C3AED;
            }

        """)


# ============================================================
# MAIN APPLICATION
# ============================================================

class PotholeDetectionUI(QMainWindow):

    TARGET_FPS = 10
    CONFIDENCE = 0.25

    def __init__(self):

        super().__init__()

        # ----------------------------------------------------
        # PATHS
        # ----------------------------------------------------

        self.BASE_DIR = Path(
            __file__
        ).resolve().parent.parent

        self.MODELS_DIR = (
            self.BASE_DIR / "models"
        )

        self.DETAILS_DIR = (
            self.BASE_DIR / "details"
        )

        self.MODELS_DIR.mkdir(
            exist_ok=True
        )

        # ----------------------------------------------------
        # MODEL
        # ----------------------------------------------------

        self.model = None
        self.MODEL_PATH = None

        # ----------------------------------------------------
        # VIDEO
        # ----------------------------------------------------

        self.cap = None
        self.video_path = None
        self.video_total_frames = 0

        # ----------------------------------------------------
        # CAMERA / VIDEO STATE
        # ----------------------------------------------------

        self.processing_camera = False
        self.processing_video = False

        # ----------------------------------------------------
        # DETECTION
        # ----------------------------------------------------

        self.last_detection_count = 0
        self.last_confidence = 0.0

        self.fps_counter = 0
        self.fps_timer = time.time()
        self.current_fps = 0.0

        # ----------------------------------------------------
        # WINDOW
        # ----------------------------------------------------

        self.setWindowTitle(
            "Pothole Detection System"
        )

        self.setMinimumSize(
            1200,
            800
        )

        # ----------------------------------------------------
        # UI
        # ----------------------------------------------------

        self.setup_ui()

        self.load_available_models()

        # ----------------------------------------------------
        # TIMER
        # ----------------------------------------------------

        self.timer = QTimer()

        self.timer.timeout.connect(
            self.process_frame
        )

        self.timer.setInterval(
            int(1000 / self.TARGET_FPS)
        )

        # ----------------------------------------------------
        # DEFAULT MODEL
        # ----------------------------------------------------

        self.check_model()

    # ========================================================
    # SETUP UI
    # ========================================================

    def setup_ui(self):

        central = QWidget()

        self.setCentralWidget(
            central
        )

        main_layout = QVBoxLayout(
            central
        )

        main_layout.setContentsMargins(
            18,
            16,
            18,
            16
        )

        main_layout.setSpacing(12)

        # ====================================================
        # HEADER
        # ====================================================

        header = QHBoxLayout()

        header_text = QVBoxLayout()
        header_text.setSpacing(2)

        title = QLabel(
            "POTHOLE DETECTION"
        )

        title.setObjectName(
            "title"
        )

        subtitle = QLabel(
            "24BEI045 & 24BEI049"
        )

        subtitle.setObjectName(
            "subtitle"
        )

        header_text.addWidget(
            title
        )

        header_text.addWidget(
            subtitle
        )

        header.addLayout(
            header_text
        )

        header.addStretch()

        self.system_status = QLabel(
            "● System ready"
        )

        self.system_status.setObjectName(
            "systemStatus"
        )

        header.addWidget(
            self.system_status
        )

        main_layout.addLayout(
            header
        )

        # ====================================================
        # MAIN CONTENT
        # ====================================================

        content_layout = QHBoxLayout()

        content_layout.setSpacing(12)

        # ====================================================
        # PREVIEW PANEL
        # ====================================================

        preview_panel = QFrame()

        preview_panel.setObjectName(
            "panel"
        )

        preview_layout = QVBoxLayout(
            preview_panel
        )

        preview_layout.setContentsMargins(
            12,
            12,
            12,
            12
        )

        preview_layout.setSpacing(8)

        # ----------------------------------------------------
        # Preview header
        # ----------------------------------------------------

        preview_header = QHBoxLayout()

        preview_title = QLabel(
            "DETECTION PREVIEW"
        )

        preview_title.setObjectName(
            "sectionTitle"
        )

        preview_header.addWidget(
            preview_title
        )

        preview_header.addStretch()

        # Zoom buttons

        zoom_out_button = QPushButton(
            "−"
        )

        zoom_out_button.setObjectName(
            "zoomButton"
        )

        zoom_out_button.clicked.connect(
            lambda: self.display.zoom_out()
        )

        self.zoom_reset_button = QPushButton(
            "100%"
        )

        self.zoom_reset_button.setObjectName(
            "zoomButton"
        )

        self.zoom_reset_button.clicked.connect(
            lambda: self.display.reset_zoom()
        )

        zoom_in_button = QPushButton(
            "+"
        )

        zoom_in_button.setObjectName(
            "zoomButton"
        )

        zoom_in_button.clicked.connect(
            lambda: self.display.zoom_in()
        )

        fit_button = QPushButton(
            "FIT"
        )

        fit_button.setObjectName(
            "zoomButton"
        )

        fit_button.clicked.connect(
            lambda: self.display.fit_image()
        )

        preview_header.addWidget(
            zoom_out_button
        )

        preview_header.addWidget(
            self.zoom_reset_button
        )

        preview_header.addWidget(
            zoom_in_button
        )

        preview_header.addWidget(
            fit_button
        )

        preview_layout.addLayout(
            preview_header
        )

        # ----------------------------------------------------
        # Display
        # ----------------------------------------------------

        self.display = ZoomableLabel()

        self.display.setObjectName(
            "display"
        )

        self.display.setText(
            "Upload an image, video or start the camera"
        )

        preview_layout.addWidget(
            self.display,
            1
        )

        # ----------------------------------------------------
        # Input buttons
        # ----------------------------------------------------

        button_layout = QHBoxLayout()

        button_layout.setSpacing(8)

        self.image_button = QPushButton(
            "UPLOAD IMAGE"
        )

        self.image_button.setObjectName(
            "actionButton"
        )

        self.image_button.clicked.connect(
            self.upload_image
        )

        self.video_button = QPushButton(
            "UPLOAD VIDEO"
        )

        self.video_button.setObjectName(
            "actionButton"
        )

        self.video_button.clicked.connect(
            self.upload_video
        )

        self.camera_button = QPushButton(
            "START CAMERA"
        )

        self.camera_button.setObjectName(
            "cameraButton"
        )

        self.camera_button.clicked.connect(
            self.start_camera
        )

        self.stop_button = QPushButton(
            "STOP"
        )

        self.stop_button.setObjectName(
            "stopButton"
        )

        self.stop_button.clicked.connect(
            self.stop_all
        )

        button_layout.addWidget(
            self.image_button,
            1
        )

        button_layout.addWidget(
            self.video_button,
            1
        )

        button_layout.addWidget(
            self.camera_button,
            1
        )

        button_layout.addWidget(
            self.stop_button,
            1
        )

        preview_layout.addLayout(
            button_layout
        )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        self.progress = QProgressBar()

        self.progress.setRange(
            0,
            100
        )

        self.progress.setValue(
            0
        )

        self.progress.setTextVisible(
            False
        )

        preview_layout.addWidget(
            self.progress
        )

        # ====================================================
        # STATS PANEL
        # ====================================================

        stats_panel = QFrame()

        stats_panel.setObjectName(
            "panel"
        )

        stats_panel.setMinimumWidth(
            240
        )

        stats_panel.setMaximumWidth(
            320
        )

        stats_layout = QVBoxLayout(
            stats_panel
        )

        stats_layout.setContentsMargins(
            12,
            12,
            12,
            12
        )

        stats_layout.setSpacing(8)

        stats_title = QLabel(
            "DETECTION DETAILS"
        )

        stats_title.setObjectName(
            "sectionTitle"
        )

        stats_layout.addWidget(
            stats_title
        )

        # ----------------------------------------------------
        # Helper
        # ----------------------------------------------------

        def create_stat_card(
            caption,
            value,
            object_name
        ):

            card = QFrame()

            card.setObjectName(
                "statCard"
            )

            layout = QVBoxLayout(
                card
            )

            layout.setContentsMargins(
                12,
                9,
                12,
                9
            )

            layout.setSpacing(2)

            caption_label = QLabel(
                caption.upper()
            )

            caption_label.setObjectName(
                "statCaption"
            )

            value_label = QLabel(
                value
            )

            value_label.setObjectName(
                object_name
            )

            layout.addWidget(
                caption_label
            )

            layout.addWidget(
                value_label
            )

            return card, value_label

        # Status

        status_card, self.status_value = (
            create_stat_card(
                "Status",
                "Ready",
                "statusValue"
            )
        )

        stats_layout.addWidget(
            status_card
        )

        # Potholes

        pothole_card, self.pothole_value = (
            create_stat_card(
                "Potholes Detected",
                "0",
                "numberValue"
            )
        )

        stats_layout.addWidget(
            pothole_card
        )

        # Confidence

        confidence_card, self.confidence_value = (
            create_stat_card(
                "Confidence",
                "0.00",
                "numberValue"
            )
        )

        stats_layout.addWidget(
            confidence_card
        )

        # FPS

        fps_card, self.fps_value = (
            create_stat_card(
                "Inference FPS",
                "0.0",
                "numberValue"
            )
        )

        stats_layout.addWidget(
            fps_card
        )

        # ----------------------------------------------------
        # MODEL
        # ----------------------------------------------------

        model_card = QFrame()

        model_card.setObjectName(
            "modelCard"
        )

        model_layout = QVBoxLayout(
            model_card
        )

        model_layout.setContentsMargins(
            12,
            9,
            12,
            9
        )

        model_layout.setSpacing(6)

        model_caption = QLabel(
            "ACTIVE MODEL"
        )

        model_caption.setObjectName(
            "statCaption"
        )

        model_layout.addWidget(
            model_caption
        )

        model_row = QHBoxLayout()

        self.model_combo = QComboBox()

        self.model_combo.setObjectName(
            "modelCombo"
        )

        self.model_combo.currentIndexChanged.connect(
            self.select_model
        )

        self.add_model_button = QPushButton(
            "+ ADD"
        )

        self.add_model_button.setObjectName(
            "addModelButton"
        )

        self.add_model_button.clicked.connect(
            self.add_model
        )

        # NEW: Settings / Model Details button

        self.details_button = QPushButton(
            "⚙"
        )

        self.details_button.setObjectName(
            "detailsButton"
        )

        self.details_button.setToolTip(
            "View training and model details"
        )

        self.details_button.clicked.connect(
            self.open_model_details
        )

        model_row.addWidget(
            self.model_combo,
            1
        )

        model_row.addWidget(
            self.add_model_button
        )

        model_row.addWidget(
            self.details_button
        )

        model_layout.addLayout(
            model_row
        )

        self.model_name_label = QLabel(
            "No model loaded"
        )

        self.model_name_label.setObjectName(
            "modelValue"
        )

        self.model_name_label.setWordWrap(
            True
        )

        model_layout.addWidget(
            self.model_name_label
        )

        stats_layout.addWidget(
            model_card
        )

        stats_layout.addStretch()

        # ====================================================
        # MAIN CONTENT ADD
        # ====================================================

        content_layout.addWidget(
            preview_panel,
            4
        )

        content_layout.addWidget(
            stats_panel,
            1
        )

        main_layout.addLayout(
            content_layout,
            1
        )

        # ====================================================
        # STYLESHEET
        # ====================================================

        self.setStyleSheet("""

            QWidget {
                background-color: #0A0B10;
                color: #F5F5F7;
                font-family: "Segoe UI";
            }

            QLabel#title {
                font-size: 30px;
                font-weight: 700;
                color: #F5F5F7;
            }

            QLabel#subtitle {
                font-size: 13px;
                color: #8B8FA3;
            }

            QLabel#systemStatus {
                font-size: 11px;
                font-weight: 600;
                color: #6EE7B7;
                padding: 9px 13px;
                background-color: #10251D;
                border: 1px solid #1E4938;
                border-radius: 8px;
            }

            QFrame#panel {
                background-color: #11131A;
                border: 1px solid #272A38;
                border-radius: 14px;
            }

            QLabel#sectionTitle {
                font-size: 11px;
                font-weight: 700;
                color: #A1A5B5;
                letter-spacing: 1px;
                padding: 2px 0px;
            }

            QLabel#display {
                background-color: #07080C;
                border: 1px solid #272A38;
                border-radius: 10px;
                color: #6F7485;
                font-size: 15px;
            }

            QFrame#statCard {
                background-color: #171923;
                border: 1px solid #252837;
                border-radius: 10px;
            }

            QFrame#statCard:hover {
                border: 1px solid #34384C;
            }

            QLabel#statCaption {
                font-size: 9px;
                font-weight: 700;
                color: #85899C;
                letter-spacing: 0.8px;
            }

            QLabel#statusValue {
                font-size: 15px;
                font-weight: 700;
                color: #FBBF24;
                padding-top: 1px;
            }

            QLabel#numberValue {
                font-size: 22px;
                font-weight: 700;
                color: #F5F5F7;
                padding-top: 1px;
            }

            QFrame#modelCard {
                background-color: #171923;
                border: 1px solid #252837;
                border-radius: 10px;
            }

            QLabel#modelValue {
                font-size: 13px;
                font-weight: 600;
                color: #D9DBE5;
            }

            QPushButton {
                min-height: 30px;
                max-height: 32px;
                border-radius: 7px;
                font-size: 10px;
                font-weight: 700;
                padding: 0px 8px;
            }

            QPushButton#actionButton {
                min-height: 30px;
                max-height: 32px;
                background-color: #171923;
                border: 1px solid #2A2D3A;
                color: #E5E7EB;
            }

            QPushButton#actionButton:hover {
                background-color: #1D1F2B;
                border: 1px solid #8B5CF6;
                color: #FFFFFF;
            }

            QPushButton#actionButton:pressed {
                background-color: #242033;
            }

            QPushButton#cameraButton {
                min-height: 30px;
                max-height: 32px;
                background-color: #12352A;
                border: 1px solid #2D8A61;
                color: #6EE7B7;
            }

            QPushButton#cameraButton:hover {
                background-color: #174332;
                border: 1px solid #34D399;
                color: #A7F3D0;
            }

            QPushButton#stopButton {
                min-height: 30px;
                max-height: 32px;
                background-color: #351B20;
                border: 1px solid #71343D;
                color: #FCA5A5;
            }

            QPushButton#stopButton:hover {
                background-color: #472126;
                border: 1px solid #F87171;
                color: #FECACA;
            }

            QPushButton#addModelButton {
                min-width: 48px;
                min-height: 34px;
                max-height: 36px;
                background-color: #8B5CF6;
                border: none;
                color: #FFFFFF;
                border-radius: 8px;
            }

            QPushButton#addModelButton:hover {
                background-color: #7C3AED;
            }

            QPushButton#detailsButton {
                min-width: 36px;
                max-width: 36px;
                min-height: 34px;
                max-height: 36px;
                background-color: #171923;
                border: 1px solid #2A2D3A;
                color: #C4B5FD;
                border-radius: 8px;
                font-size: 15px;
            }

            QPushButton#detailsButton:hover {
                background-color: #211C31;
                border: 1px solid #8B5CF6;
                color: #FFFFFF;
            }

            QComboBox#modelCombo {
                min-height: 34px;
                max-height: 36px;
                background-color: #171923;
                border: 1px solid #2A2D3A;
                border-radius: 8px;
                padding: 0px 10px;
                color: #E5E7EB;
                font-size: 11px;
                font-weight: 600;
            }

            QComboBox#modelCombo:hover {
                border: 1px solid #8B5CF6;
            }

            QComboBox#modelCombo::drop-down {
                border: none;
                width: 28px;
            }

            QComboBox#modelCombo QAbstractItemView {
                background-color: #171923;
                color: #E5E7EB;
                border: 1px solid #2A2D3A;
                selection-background-color: #29223F;
                selection-color: #FFFFFF;
            }

            QPushButton#zoomButton {
                min-width: 30px;
                max-width: 42px;
                min-height: 26px;
                max-height: 26px;
                padding: 0px 5px;
                border-radius: 6px;
                background-color: #171923;
                border: 1px solid #2A2D3A;
                color: #C9CBD6;
                font-size: 9px;
            }

            QPushButton#zoomButton:hover {
                background-color: #211C31;
                border: 1px solid #8B5CF6;
                color: #FFFFFF;
            }

            QProgressBar {
                height: 10px;
                border-radius: 5px;
                background-color: #171923;
                border: 1px solid #222532;
                text-align: center;
                color: #D5D7E0;
                font-size: 8px;
            }

            QProgressBar::chunk {
                background-color: #8B5CF6;
                border-radius: 5px;
            }

        """)

    # ========================================================
    # MODEL MANAGEMENT
    # ========================================================

    def load_available_models(self):

        self.model_combo.blockSignals(
            True
        )

        self.model_combo.clear()

        model_files = sorted(
            self.MODELS_DIR.glob("*.pt")
        )

        for model_file in model_files:

            self.model_combo.addItem(
                model_file.name,
                str(model_file)
            )

        self.model_combo.blockSignals(
            False
        )

        # Prefer best.pt

        best_index = self.model_combo.findText(
            "best.pt"
        )

        if best_index >= 0:

            self.model_combo.setCurrentIndex(
                best_index
            )

        elif self.model_combo.count() > 0:

            self.model_combo.setCurrentIndex(
                0
            )

    # ========================================================
    # ADD MODEL
    # ========================================================

    def add_model(self):

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select YOLO Model",
            "",
            "YOLO Model (*.pt)"
        )

        if not file_path:
            return

        source = Path(file_path)

        destination = (
            self.MODELS_DIR /
            source.name
        )

        try:

            if source.resolve() != destination.resolve():

                destination.write_bytes(
                    source.read_bytes()
                )

            self.load_available_models()

            index = self.model_combo.findText(
                destination.name
            )

            if index >= 0:

                self.model_combo.setCurrentIndex(
                    index
                )

        except Exception as e:

            QMessageBox.critical(
                self,
                "Model Error",
                f"Could not add model:\n\n{e}"
            )

    # ========================================================
    # SELECT MODEL
    # ========================================================

    def select_model(self, index):

        if index < 0:
            return

        model_path = self.model_combo.itemData(
            index
        )

        if not model_path:
            return

        self.stop_all()

        try:

            self.model = YOLO(
                model_path
            )

            self.MODEL_PATH = Path(
                model_path
            )

            self.model_name_label.setText(
                self.MODEL_PATH.name
            )

            self.system_status.setText(
                "● MODEL LOADED"
            )

            self.status_value.setText(
                "Ready"
            )

        except Exception as e:

            self.model = None
            self.MODEL_PATH = None

            QMessageBox.critical(
                self,
                "Model Error",
                f"Could not load model:\n\n{e}"
            )

    # ========================================================
    # CHECK MODEL
    # ========================================================

    def check_model(self):

        if self.model_combo.count() == 0:

            self.system_status.setText(
                "● NO MODEL"
            )

            self.status_value.setText(
                "No model"
            )

            return

        if self.model is None:

            index = self.model_combo.currentIndex()

            if index >= 0:
                self.select_model(index)

    # ========================================================
    # OPEN MODEL DETAILS
    # ========================================================

    def open_model_details(self):

        if self.MODEL_PATH is None:

            QMessageBox.warning(
                self,
                "No Model",
                "Please load a model first."
            )

            return

        if not self.DETAILS_DIR.exists():

            QMessageBox.warning(
                self,
                "Training Details Not Found",
                "The 'details' folder was not found."
            )

            return

        dialog = ModelDetailsDialog(
            self.MODEL_PATH,
            self.DETAILS_DIR,
            self.model,
            self
        )

        dialog.exec()

    # ========================================================
    # IMAGE UPLOAD
    # ========================================================

    def upload_image(self):

        self.stop_all()

        if self.model is None:

            QMessageBox.warning(
                self,
                "No Model",
                "Please load a YOLO model first."
            )

            return

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image",
            "",
            "Images (*.jpg *.jpeg *.png *.bmp *.webp)"
        )

        if not file_path:
            return

        image = cv2.imread(
            file_path
        )

        if image is None:

            QMessageBox.warning(
                self,
                "Image Error",
                "Could not read the selected image."
            )

            return

        self.status_value.setText(
            "Detecting"
        )

        start = time.time()

        try:

            results = self.model.predict(
                image,
                conf=self.CONFIDENCE,
                verbose=False
            )

            result = results[0]

            annotated = result.plot()

            elapsed = time.time() - start

            self.current_fps = (
                1.0 / elapsed
                if elapsed > 0
                else 0
            )

            count = 0
            confidence = 0.0

            if result.boxes is not None:

                count = len(
                    result.boxes
                )

                if count > 0:

                    confidence = float(
                        result.boxes.conf.max()
                    )

            self.update_detection_stats(
                count,
                confidence
            )

            self.display_image(
                annotated
            )

            self.status_value.setText(
                "Detection Complete"
            )

        except Exception as e:

            QMessageBox.critical(
                self,
                "Detection Error",
                f"Detection failed:\n\n{e}"
            )

    # ========================================================
    # VIDEO UPLOAD
    # ========================================================

    def upload_video(self):

        self.stop_all()

        if self.model is None:

            QMessageBox.warning(
                self,
                "No Model",
                "Please load a YOLO model first."
            )

            return

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video",
            "",
            "Videos (*.mp4 *.avi *.mov *.mkv)"
        )

        if not file_path:
            return

        self.cap = cv2.VideoCapture(
            file_path
        )

        if not self.cap.isOpened():

            QMessageBox.warning(
                self,
                "Video Error",
                "Could not open the selected video."
            )

            self.cap = None

            return

        self.video_path = file_path

        self.video_total_frames = int(
            self.cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        self.processing_video = True
        self.processing_camera = False

        self.status_value.setText(
            "Processing Video"
        )

        self.timer.start()

    # ========================================================
    # CAMERA
    # ========================================================

    def start_camera(self):

        self.stop_all()

        if self.model is None:

            QMessageBox.warning(
                self,
                "No Model",
                "Please load a YOLO model first."
            )

            return

        self.cap = cv2.VideoCapture(
            0
        )

        if not self.cap.isOpened():

            QMessageBox.warning(
                self,
                "Camera Error",
                "Could not access the camera."
            )

            self.cap = None

            return

        self.processing_camera = True
        self.processing_video = False

        self.status_value.setText(
            "Camera Active"
        )

        self.timer.start()

    # ========================================================
    # PROCESS FRAME
    # ========================================================

    def process_frame(self):

        if self.cap is None:
            return

        ret, frame = self.cap.read()

        if not ret:

            self.stop_video()

            return

        start = time.time()

        try:

            results = self.model.predict(
                frame,
                conf=self.CONFIDENCE,
                verbose=False
            )

            result = results[0]

            annotated = result.plot()

            elapsed = time.time() - start

            if elapsed > 0:
                self.current_fps = (
                    1.0 / elapsed
                )

            count = 0
            confidence = 0.0

            if result.boxes is not None:

                count = len(
                    result.boxes
                )

                if count > 0:

                    confidence = float(
                        result.boxes.conf.max()
                    )

            self.update_detection_stats(
                count,
                confidence
            )

            self.display_image(
                annotated
            )

            # ------------------------------------------------
            # Video progress
            # ------------------------------------------------

            if (
                self.processing_video
                and self.video_total_frames > 0
            ):

                current_frame = int(
                    self.cap.get(
                        cv2.CAP_PROP_POS_FRAMES
                    )
                )

                progress = int(
                    (
                        current_frame /
                        self.video_total_frames
                    ) * 100
                )

                self.progress.setValue(
                    max(
                        0,
                        min(
                            progress,
                            100
                        )
                    )
                )

        except Exception as e:

            print(
                "Detection error:",
                e
            )

    # ========================================================
    # DISPLAY IMAGE
    # ========================================================

    def display_image(self, image):

        if image is None:
            return

        rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        h, w, ch = rgb.shape

        bytes_per_line = ch * w

        qimage = QImage(
            rgb.data,
            w,
            h,
            bytes_per_line,
            QImage.Format_RGB888
        ).copy()

        pixmap = QPixmap.fromImage(
            qimage
        )

        self.display.set_image(
            pixmap
        )

    # ========================================================
    # UPDATE STATS
    # ========================================================

    def update_detection_stats(
        self,
        count,
        confidence
    ):

        self.last_detection_count = count
        self.last_confidence = confidence

        self.pothole_value.setText(
            str(count)
        )

        self.confidence_value.setText(
            f"{confidence:.2f}"
        )

        self.fps_value.setText(
            f"{self.current_fps:.1f}"
        )

        if count > 0:

            self.status_value.setText(
                "Pothole Detected"
            )

        else:

            self.status_value.setText(
                "Road Clear"
            )

    # ========================================================
    # STOP VIDEO
    # ========================================================

    def stop_video(self):

        self.timer.stop()

        if self.cap is not None:

            self.cap.release()

            self.cap = None

        self.processing_video = False
        self.processing_camera = False

        self.progress.setValue(
            0
        )

        self.status_value.setText(
            "Ready"
        )

    # ========================================================
    # STOP EVERYTHING
    # ========================================================

    def stop_all(self):

        self.stop_video()

    # ========================================================
    # CLOSE
    # ========================================================

    def closeEvent(self, event):

        self.stop_all()

        event.accept()


# ============================================================
# APPLICATION
# ============================================================

if __name__ == "__main__":

    app = QApplication(
        sys.argv
    )

    window = PotholeDetectionUI()

    window.show()

    sys.exit(
        app.exec()
    )
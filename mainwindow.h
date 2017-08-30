#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "activation_funcs_worker.h"

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();
    QMap<QString,activation_funcs_type> *string_to_activation_funcs_type_dictionary;

private slots:
    void on_show_btn_clicked();
    void newGraphic(void);

private:
    Ui::MainWindow *ui;
};

#endif // MAINWINDOW_H

#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <math.h>


MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    string_to_activation_funcs_type_dictionary = new QMap<QString,activation_funcs_type>();
    string_to_activation_funcs_type_dictionary->insert("sigmoidal",activation_funcs_type::SIGM);
    string_to_activation_funcs_type_dictionary->insert("tangencial", activation_funcs_type::THN);
    string_to_activation_funcs_type_dictionary->insert("relu", activation_funcs_type::RELU);
    string_to_activation_funcs_type_dictionary->insert("softmax", activation_funcs_type::SOFTMAX);

    for(auto elem : string_to_activation_funcs_type_dictionary->keys())
        {
            ui->func_name_comboBox->addItem(elem);
        }
}

MainWindow::~MainWindow()
{
    delete ui;
}



int GetNod(int ch,int zn) {
 return (ch ? GetNod(zn%ch,ch) : zn);
 }


QString ClipFraction (int ch,int zn) {
 int nod = GetNod(ch,zn);
 if (nod>1) { ch/=nod; zn/=nod; }
 QString result;
 if (ch==0) result.setNum(0);
 else if (ch==zn) result.setNum(1);
 else if (zn==1) result.setNum(ch);
 else result = QString("%1/%2").arg(ch).arg(zn);
 return result;
}

void MainWindow::newGraphic()
{
        //input data configuration
        int points_num= ui->points_num_spinBox->value();
        // data bounds
        double x_min,x_max,y_min,y_max;
        x_min= ui->min_val_box->value();
        x_max= ui->max_val_box->value();
        //data
        QVector <double> x(points_num),y(points_num);

        std::shared_ptr<std::vector<double>> test_inputs = std::make_shared<std::vector<double>>(points_num);
        std::shared_ptr<std::vector<double>> test_results = std::make_shared<std::vector<double>>(points_num);

        double point_step = (x_max-x_min)/double(points_num-1);
            for(int i=0; i<points_num;++i)
            {
                test_inputs->at(i)=x_min+point_step*(double)(i);
            }
    //choose the function
        activation_funcs_type cur_activation_funcs_type = string_to_activation_funcs_type_dictionary->value(ui->func_name_comboBox->currentText());
        std::shared_ptr<activation_funcs_worker> af = activation_funcs_worker::create_activation_funcs_worker(cur_activation_funcs_type);
    //apply the function
        if(ui->is_derived_checkBox->checkState()==Qt::Checked)
            {af->apply_d(*test_inputs,*test_results);}
        else
            {af->apply(*test_inputs,*test_results);}

    //copy std::vector to QVector

        for (int i=0; i<points_num; i++)
        {
         x[i] = test_inputs->at(i);
         y[i] = test_results->at(i);
        }
    //find min and max y-values if required. min and max x-settings are given by input data settings.
        if(ui->autoscale_checkBox->isChecked())
            {
                y_min=test_results->at(0);
                y_max=test_results->at(0);

                for (int i=0; i<points_num; i++)
                    {
                        if(y[i]>y_max)
                        y_max=y[i];
                        if(y[i]<y_min)
                        y_min=y[i];
                    }
        }
        else
        {
            x_min=ui->x_min_box->value();
            x_max=ui->x_max_box->value();
            y_min=ui->y_min_box->value();
            y_max=ui->y_max_box->value();
        }

    //setting data
    ui->GraphicWidget->addGraph();
    ui->GraphicWidget->graph(0)->setData(x,y);

    //set axis and labels
    ui->GraphicWidget->xAxis->setLabel("x");
    ui->GraphicWidget->xAxis->setRange(x_min,x_max); //1

    ui->GraphicWidget->yAxis->setLabel("act_f(x)");
    ui->GraphicWidget->yAxis->setRange(y_min,y_max);
    //show plot
    ui->GraphicWidget->replot();
}

void MainWindow::on_show_btn_clicked()
{
    newGraphic();
}

package com.nflabs.zeppelin.clustering;

import java.io.IOException;
import java.io.InputStream;
import java.io.StringWriter;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.Random;

import org.apache.commons.io.IOUtils;
import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.clustering.KMeans;
import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.scheduler.Pool;
import org.apache.zeppelin.helium.Application;
import org.apache.zeppelin.helium.ApplicationArgument;
import org.apache.zeppelin.helium.ApplicationException;
import org.apache.zeppelin.helium.Signal;
import org.apache.zeppelin.interpreter.InterpreterContext;
import org.apache.zeppelin.interpreter.InterpreterContextRunner;
import org.apache.zeppelin.interpreter.InterpreterResult;
import org.apache.zeppelin.interpreter.InterpreterResult.Code;
import org.apache.zeppelin.interpreter.data.ColumnDef;
import org.apache.zeppelin.interpreter.data.TableData;
import org.apache.zeppelin.interpreter.dev.ZeppelinApplicationDevServer;
import org.apache.zeppelin.resource.ResourceInfo;
import org.apache.zeppelin.resource.ResourceKey;
import org.apache.zeppelin.resource.WellKnownResource;


public class Clustering extends Application {

  private SparkContext sc;
  private JavaSparkContext jsc;
  private InterpreterContext context;
  private TableData tableData;

  @Override
  protected void onChange(String name, Object oldObject, Object newObject) {
    System.err.println("Change " + name + " : " + oldObject + " -> " + newObject);
    if (name.equals("run")) {
      int numClusters = Integer.parseInt(this.get(context, "numCluster").toString());
      int numIterations = Integer.parseInt(this.get(context, "iteration").toString());


      int col = -1;
      // find first numeric column
      ColumnDef[] columnDef = tableData.getColumnDef();


      for (int c = 0; c < columnDef.length; c++) {
        try {
          Float.parseFloat((String) tableData.getData(0, c));
          col = c;
          break;
        } catch (Exception e) {
          continue;
        }
      }

      if (col == -1) {
        return;
      }

      LinkedList<Vector> vectors = new LinkedList<Vector>();

      for (int i = 0 ; i < tableData.length(); i++) {
        double val = Double.parseDouble(tableData.getData(i, col).toString());
        vectors.add(Vectors.dense(val));
      }

      JavaRDD<Vector> inputVector = jsc.parallelize(vectors).cache();
      KMeansModel clusters = KMeans.train(inputVector.rdd(), numClusters, numIterations);

      StringWriter msg = new StringWriter();
      // header
      for (int c = 0; c < columnDef.length; c++) {
        msg.write(columnDef[c].getName() + "\t");
      }
      msg.write("cluster\n");

      for (int r = 0; r < tableData.length(); r++) {
        for (int c = 0; c < columnDef.length; c++) {
          msg.write(tableData.getData(r, c).toString() + "\t");
        }
        msg.write(clusters.predict(
            Vectors.dense(Double.parseDouble(tableData.getData(r, col).toString()))) + "\n");
      }

      this.put(context, "result", msg.toString());
      this.put(context, "run", "idle");
    }
  }


  @Override
  public void signal(Signal signal) {

  }

  @Override
  public void load() throws ApplicationException, IOException {

  }

  JavaRDD<Vector> toVector(TableData tableData) {
    int col = -1;
    // find first numeric column
    ColumnDef[] columnDef = tableData.getColumnDef();


    for (int c = 0; c < columnDef.length; c++) {
      try {
        Float.parseFloat((String) tableData.getData(0, c));
        col = c;
        break;
      } catch (Exception e) {
        continue;
      }
    }

    if (col == -1) {
      return null;
    }

    LinkedList<Vector> vectors = new LinkedList<Vector>();

    for (int i = 0 ; i < tableData.length(); i++) {
      double val = Double.parseDouble(tableData.getData(i, col).toString());
      vectors.add(Vectors.dense(val));
    }
    return jsc.parallelize(vectors);
  }

  @Override
  public void run(ApplicationArgument arg, InterpreterContext context) throws ApplicationException,
      IOException {

    // load resource from classpath
    context.out.writeResource("clustering/Clustering.html");

    this.put(context, "numCluster", 3);
    this.put(context, "iteration", 10);
    this.put(context, "run", "idle");
    this.watch(context, "run");
    this.context = context;

    // get TableData
    tableData = (TableData) context.getResourcePool().get(
        arg.getResource().location(), arg.getResource().name());


    // get spark context
    Collection<ResourceInfo> infos = context.getResourcePool().search(
        WellKnownResource.SPARK_CONTEXT.type() + ".*");
    if (infos == null || infos.size() == 0) {
      throw new ApplicationException("SparkContext not available");
    }

    Iterator<ResourceInfo> it = infos.iterator();
    while (it.hasNext()) {
      ResourceInfo info = it.next();
      sc = (SparkContext) context.getResourcePool().get(info.name());
      if (sc != null) {
        break;
      }
    }

    jsc = new JavaSparkContext(sc);

  }

  @Override
  public void unload() throws ApplicationException, IOException {

  }


  private static String generateData() throws IOException {
    InputStream ins = ClassLoader.getSystemResourceAsStream("clustering/mockdata.txt");
    String data = IOUtils.toString(ins);
    return data;
  }

  /**
   * Development mode
   * @param args
   * @throws Exception
   */
  public static void main(String [] args) throws Exception {
    // create development server
    ZeppelinApplicationDevServer dev = new ZeppelinApplicationDevServer(Clustering.class.getName());

    TableData tableData = new TableData(new InterpreterResult(Code.SUCCESS, generateData()));

    dev.server.getResourcePool().put("tabledata", tableData);

    // set application argument
    ApplicationArgument arg = new ApplicationArgument(new ResourceKey(
        dev.server.getResourcePoolId(),
        "tabledata"
        ));
    dev.setArgument(arg);


    // set sparkcontext
    // create spark conf
    SparkConf conf = new SparkConf();
    conf.setMaster("local[*]");
    conf.setAppName("Clustering");

    // create spark context
    SparkContext sc = new SparkContext(conf);
    dev.server.getResourcePool().put(WellKnownResource.SPARK_CONTEXT.type() + "#aaa", sc);

    // start
    dev.server.start();
    dev.server.join();
  }
}



import scala.util.Random
import com.github.fommil.netlib.BLAS.{ getInstance => blas }
import org.apache.spark.rdd._
import org.apache.spark.graphx._
import com.github.fommil.netlib.BLAS.{getInstance => blas}
object MF {
  class Conf(
    var rank: Int,
    var maxIters: Int,
    var minVal: Double,
    var maxVal: Double,
    var lambda: Double,
    var gamma: Double)
      extends Serializable
      
  def run(edges: RDD[Edge[Double]], conf: Conf): Graph[(Array[Double], Double), Double] =
    {
      
      def defaultF(rank: Int): (Array[Double], Double) = {
        val v1 = Array.fill(rank)(Random.nextDouble())
        (v1, 0.0)
      }

      edges.cache()
      var g = Graph.fromEdges(edges, defaultF(conf.rank)).cache()
      materialize(g)
      edges.unpersist()

      def sendMsgTrainF(conf: Conf)(ctx: EdgeContext[(Array[Double], Double), Double, Array[Double]]) {
 
        val (p, q) = (ctx.srcAttr._1, ctx.dstAttr._1)
        val rank = p.length
        var pred = blas.ddot(rank, q, 1, p, 1)
        pred = math.max(pred, conf.minVal)
        pred = math.min(pred, conf.maxVal)
        val err = ctx.attr - pred
        val updateP = q.clone()
        blas.dscal(rank, err * conf.gamma, updateP, 1)
        blas.daxpy(rank, -conf.gamma * conf.lambda, p, 1, updateP, 1)
        updateQ = (err * p - conf.lambda * q) * conf.gamma
        val updateQ = p.clone()
        blas.dscal(rank, err * conf.gamma, updateQ, 1)
        blas.daxpy(rank, -conf.gamma * conf.lambda, q, 1, updateQ, 1)
        ctx.sendToSrc(updateP)
        ctx.sendToDst(updateQ)
      }

      for (i <- 0 until conf.maxIters) {
         
        g.cache()
        val t2 = g.aggregateMessages(
          sendMsgTrainF(conf),
          (g1: Array[Double], g2: Array[Double]) =>
            {
              val out1 = g1.clone()
              blas.daxpy(out1.length, 1.0, g2, 1, out1, 1)
              out1
            })
        val gJoinT2 = g.outerJoinVertices(t2) {
          (vid: VertexId,
          vd: (Array[Double], Double),
          msg: Option[Array[Double]]) =>
            {
              val out1 = vd._1.clone()
              blas.daxpy(out1.length, 1.0, msg.get, 1, out1, 1)
              (out1, 0.0)
            }
        }.cache()
        materialize(gJoinT2)
        g.unpersist()
        g = gJoinT2
      }

      def sendMsgTestF(conf: Conf)(ctx: EdgeContext[(Array[Double], Double), Double, Double]) {
        
        val (p, q) = (ctx.srcAttr._1, ctx.dstAttr._1)
        var pred = blas.ddot(q.length, q, 1, p, 1)
        pred = math.max(pred, conf.minVal)
        pred = math.min(pred, conf.maxVal)
        val err = (ctx.attr - pred) * (ctx.attr - pred)
        ctx.sendToDst(err)
      }

      g.cache()
      val t3 = g.aggregateMessages[Double](sendMsgTestF(conf), _ + _)
      val gJoinT3 = g.outerJoinVertices(t3) {
        (vid: VertexId, vd: (Array[Double], Double), msg: Option[Double]) =>
          if (msg.isDefined) (vd._1, msg.get) else vd
      }.cache()
      materialize(gJoinT3)
      g.unpersist()
      g = gJoinT3
 
      val newVertices = g.vertices.mapValues(v => (v._1.toArray, v._2.toDouble))
      Graph(newVertices, g.edges)
    }

  private def materialize(g: Graph[_, _]): Unit = {
    g.vertices.count()
    g.edges.count()
  }
}
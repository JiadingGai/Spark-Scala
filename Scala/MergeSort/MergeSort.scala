object MergeSort {

  implicit def IntIntLessThan(x: Int, y: Int) = x < y
  
  def mergeSort[T](xs: List[T])(implicit pred: (T, T) => Boolean): List[T] = {
    val m = xs.length / 2
    if (m == 0) 
      xs
    else {
      @scala.annotation.tailrec
      def merge(ls: List[T], rs: List[T], acc: List[T] = List()): List[T] = (ls, rs) match {
        case (Nil, _) => acc ++ rs
        case (_, Nil) => acc ++ ls
        case (l :: ls1, r :: rs1) =>
          if (pred(l, r))
            merge(ls1, rs, acc :+ l)
          else
            merge(ls, rs1, acc :+ r)
      }
  
      val (l, r) = xs splitAt m
      merge(mergeSort(l), mergeSort(r))
    }
  }
  
  def main(args: Array[String]) {
    println(mergeSort(List(4, 2, 9, 7, 6, 5, 1, 3, 8)))
  }
}

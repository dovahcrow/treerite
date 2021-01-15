(function() {var implementors = {};
implementors["ndarray"] = [{"text":"impl&lt;A, B, S, S2, D, E&gt; BitOr&lt;ArrayBase&lt;S2, E&gt;&gt; for ArrayBase&lt;S, D&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;A: Clone + BitOr&lt;B, Output = A&gt;,<br>&nbsp;&nbsp;&nbsp;&nbsp;B: Clone,<br>&nbsp;&nbsp;&nbsp;&nbsp;S: DataOwned&lt;Elem = A&gt; + DataMut,<br>&nbsp;&nbsp;&nbsp;&nbsp;S2: Data&lt;Elem = B&gt;,<br>&nbsp;&nbsp;&nbsp;&nbsp;D: Dimension,<br>&nbsp;&nbsp;&nbsp;&nbsp;E: Dimension,&nbsp;</span>","synthetic":false,"types":[]},{"text":"impl&lt;'a, A, B, S, S2, D, E&gt; BitOr&lt;&amp;'a ArrayBase&lt;S2, E&gt;&gt; for ArrayBase&lt;S, D&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;A: Clone + BitOr&lt;B, Output = A&gt;,<br>&nbsp;&nbsp;&nbsp;&nbsp;B: Clone,<br>&nbsp;&nbsp;&nbsp;&nbsp;S: DataOwned&lt;Elem = A&gt; + DataMut,<br>&nbsp;&nbsp;&nbsp;&nbsp;S2: Data&lt;Elem = B&gt;,<br>&nbsp;&nbsp;&nbsp;&nbsp;D: Dimension,<br>&nbsp;&nbsp;&nbsp;&nbsp;E: Dimension,&nbsp;</span>","synthetic":false,"types":[]},{"text":"impl&lt;'a, A, B, S, S2, D, E&gt; BitOr&lt;&amp;'a ArrayBase&lt;S2, E&gt;&gt; for &amp;'a ArrayBase&lt;S, D&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;A: Clone + BitOr&lt;B, Output = A&gt;,<br>&nbsp;&nbsp;&nbsp;&nbsp;B: Clone,<br>&nbsp;&nbsp;&nbsp;&nbsp;S: Data&lt;Elem = A&gt;,<br>&nbsp;&nbsp;&nbsp;&nbsp;S2: Data&lt;Elem = B&gt;,<br>&nbsp;&nbsp;&nbsp;&nbsp;D: Dimension,<br>&nbsp;&nbsp;&nbsp;&nbsp;E: Dimension,&nbsp;</span>","synthetic":false,"types":[]},{"text":"impl&lt;A, S, D, B&gt; BitOr&lt;B&gt; for ArrayBase&lt;S, D&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;A: Clone + BitOr&lt;B, Output = A&gt;,<br>&nbsp;&nbsp;&nbsp;&nbsp;S: DataOwned&lt;Elem = A&gt; + DataMut,<br>&nbsp;&nbsp;&nbsp;&nbsp;D: Dimension,<br>&nbsp;&nbsp;&nbsp;&nbsp;B: ScalarOperand,&nbsp;</span>","synthetic":false,"types":[]},{"text":"impl&lt;'a, A, S, D, B&gt; BitOr&lt;B&gt; for &amp;'a ArrayBase&lt;S, D&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;A: Clone + BitOr&lt;B, Output = A&gt;,<br>&nbsp;&nbsp;&nbsp;&nbsp;S: Data&lt;Elem = A&gt;,<br>&nbsp;&nbsp;&nbsp;&nbsp;D: Dimension,<br>&nbsp;&nbsp;&nbsp;&nbsp;B: ScalarOperand,&nbsp;</span>","synthetic":false,"types":[]},{"text":"impl&lt;S, D&gt; BitOr&lt;ArrayBase&lt;S, D&gt;&gt; for i8 <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;S: DataOwned&lt;Elem = i8&gt; + DataMut,<br>&nbsp;&nbsp;&nbsp;&nbsp;D: Dimension,&nbsp;</span>","synthetic":false,"types":[]},{"text":"impl&lt;'a, S, D&gt; BitOr&lt;&amp;'a ArrayBase&lt;S, D&gt;&gt; for i8 <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;S: Data&lt;Elem = i8&gt;,<br>&nbsp;&nbsp;&nbsp;&nbsp;D: Dimension,&nbsp;</span>","synthetic":false,"types":[]},{"text":"impl&lt;S, D&gt; BitOr&lt;ArrayBase&lt;S, D&gt;&gt; for u8 <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;S: DataOwned&lt;Elem = u8&gt; + DataMut,<br>&nbsp;&nbsp;&nbsp;&nbsp;D: Dimension,&nbsp;</span>","synthetic":false,"types":[]},{"text":"impl&lt;'a, S, D&gt; BitOr&lt;&amp;'a ArrayBase&lt;S, D&gt;&gt; for u8 <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;S: Data&lt;Elem = u8&gt;,<br>&nbsp;&nbsp;&nbsp;&nbsp;D: Dimension,&nbsp;</span>","synthetic":false,"types":[]},{"text":"impl&lt;S, D&gt; BitOr&lt;ArrayBase&lt;S, D&gt;&gt; for i16 <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;S: DataOwned&lt;Elem = i16&gt; + DataMut,<br>&nbsp;&nbsp;&nbsp;&nbsp;D: Dimension,&nbsp;</span>","synthetic":false,"types":[]},{"text":"impl&lt;'a, S, D&gt; BitOr&lt;&amp;'a ArrayBase&lt;S, D&gt;&gt; for i16 <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;S: Data&lt;Elem = i16&gt;,<br>&nbsp;&nbsp;&nbsp;&nbsp;D: Dimension,&nbsp;</span>","synthetic":false,"types":[]},{"text":"impl&lt;S, D&gt; BitOr&lt;ArrayBase&lt;S, D&gt;&gt; for u16 <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;S: DataOwned&lt;Elem = u16&gt; + DataMut,<br>&nbsp;&nbsp;&nbsp;&nbsp;D: Dimension,&nbsp;</span>","synthetic":false,"types":[]},{"text":"impl&lt;'a, S, D&gt; BitOr&lt;&amp;'a ArrayBase&lt;S, D&gt;&gt; for u16 <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;S: Data&lt;Elem = u16&gt;,<br>&nbsp;&nbsp;&nbsp;&nbsp;D: Dimension,&nbsp;</span>","synthetic":false,"types":[]},{"text":"impl&lt;S, D&gt; BitOr&lt;ArrayBase&lt;S, D&gt;&gt; for i32 <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;S: DataOwned&lt;Elem = i32&gt; + DataMut,<br>&nbsp;&nbsp;&nbsp;&nbsp;D: Dimension,&nbsp;</span>","synthetic":false,"types":[]},{"text":"impl&lt;'a, S, D&gt; BitOr&lt;&amp;'a ArrayBase&lt;S, D&gt;&gt; for i32 <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;S: Data&lt;Elem = i32&gt;,<br>&nbsp;&nbsp;&nbsp;&nbsp;D: Dimension,&nbsp;</span>","synthetic":false,"types":[]},{"text":"impl&lt;S, D&gt; BitOr&lt;ArrayBase&lt;S, D&gt;&gt; for u32 <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;S: DataOwned&lt;Elem = u32&gt; + DataMut,<br>&nbsp;&nbsp;&nbsp;&nbsp;D: Dimension,&nbsp;</span>","synthetic":false,"types":[]},{"text":"impl&lt;'a, S, D&gt; BitOr&lt;&amp;'a ArrayBase&lt;S, D&gt;&gt; for u32 <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;S: Data&lt;Elem = u32&gt;,<br>&nbsp;&nbsp;&nbsp;&nbsp;D: Dimension,&nbsp;</span>","synthetic":false,"types":[]},{"text":"impl&lt;S, D&gt; BitOr&lt;ArrayBase&lt;S, D&gt;&gt; for i64 <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;S: DataOwned&lt;Elem = i64&gt; + DataMut,<br>&nbsp;&nbsp;&nbsp;&nbsp;D: Dimension,&nbsp;</span>","synthetic":false,"types":[]},{"text":"impl&lt;'a, S, D&gt; BitOr&lt;&amp;'a ArrayBase&lt;S, D&gt;&gt; for i64 <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;S: Data&lt;Elem = i64&gt;,<br>&nbsp;&nbsp;&nbsp;&nbsp;D: Dimension,&nbsp;</span>","synthetic":false,"types":[]},{"text":"impl&lt;S, D&gt; BitOr&lt;ArrayBase&lt;S, D&gt;&gt; for u64 <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;S: DataOwned&lt;Elem = u64&gt; + DataMut,<br>&nbsp;&nbsp;&nbsp;&nbsp;D: Dimension,&nbsp;</span>","synthetic":false,"types":[]},{"text":"impl&lt;'a, S, D&gt; BitOr&lt;&amp;'a ArrayBase&lt;S, D&gt;&gt; for u64 <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;S: Data&lt;Elem = u64&gt;,<br>&nbsp;&nbsp;&nbsp;&nbsp;D: Dimension,&nbsp;</span>","synthetic":false,"types":[]},{"text":"impl&lt;S, D&gt; BitOr&lt;ArrayBase&lt;S, D&gt;&gt; for i128 <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;S: DataOwned&lt;Elem = i128&gt; + DataMut,<br>&nbsp;&nbsp;&nbsp;&nbsp;D: Dimension,&nbsp;</span>","synthetic":false,"types":[]},{"text":"impl&lt;'a, S, D&gt; BitOr&lt;&amp;'a ArrayBase&lt;S, D&gt;&gt; for i128 <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;S: Data&lt;Elem = i128&gt;,<br>&nbsp;&nbsp;&nbsp;&nbsp;D: Dimension,&nbsp;</span>","synthetic":false,"types":[]},{"text":"impl&lt;S, D&gt; BitOr&lt;ArrayBase&lt;S, D&gt;&gt; for u128 <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;S: DataOwned&lt;Elem = u128&gt; + DataMut,<br>&nbsp;&nbsp;&nbsp;&nbsp;D: Dimension,&nbsp;</span>","synthetic":false,"types":[]},{"text":"impl&lt;'a, S, D&gt; BitOr&lt;&amp;'a ArrayBase&lt;S, D&gt;&gt; for u128 <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;S: Data&lt;Elem = u128&gt;,<br>&nbsp;&nbsp;&nbsp;&nbsp;D: Dimension,&nbsp;</span>","synthetic":false,"types":[]},{"text":"impl&lt;S, D&gt; BitOr&lt;ArrayBase&lt;S, D&gt;&gt; for bool <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;S: DataOwned&lt;Elem = bool&gt; + DataMut,<br>&nbsp;&nbsp;&nbsp;&nbsp;D: Dimension,&nbsp;</span>","synthetic":false,"types":[]},{"text":"impl&lt;'a, S, D&gt; BitOr&lt;&amp;'a ArrayBase&lt;S, D&gt;&gt; for bool <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;S: Data&lt;Elem = bool&gt;,<br>&nbsp;&nbsp;&nbsp;&nbsp;D: Dimension,&nbsp;</span>","synthetic":false,"types":[]}];
if (window.register_implementors) {window.register_implementors(implementors);} else {window.pending_implementors = implementors;}})()
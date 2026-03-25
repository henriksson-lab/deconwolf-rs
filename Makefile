all:
	cargo run 

gitaddall:
	git add src ui assets

loc:
	find src tests -name '*.rs' | xargs wc -l

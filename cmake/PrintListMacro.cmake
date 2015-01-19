#Print macro for a list
MACRO(PRINTLIST MYCOMMENT MYLIST)
message(STATUS "${MYCOMMENT}")
foreach(dir ${MYLIST})
	message(STATUS "      ${dir}")
endforeach(dir)
endmacro(PRINTLIST)





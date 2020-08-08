class BackTracing:
    def __init__(self, bo):
        self.bo = bo 

    def solve(self):
        find = self.find_empty()
        if not find:
            return True
        else:
            row, col = find

        for i in range(1,10):
            if self.valid(i, (row, col)):
                self.bo[row][col] = i

                if self.solve():
                    return True

                self.bo[row][col] = 0

        return False


    def valid(self, num, pos):
        # Check row
        for i in range(len(self.bo[0])):
            if self.bo[pos[0]][i] == num and pos[1] != i:
                return False

        # Check column
        for i in range(len(self.bo)):
            if self.bo[i][pos[1]] == num and pos[0] != i:
                return False

        # Check box
        box_x = pos[1] // 3
        box_y = pos[0] // 3

        for i in range(box_y*3, box_y*3 + 3):
            for j in range(box_x * 3, box_x*3 + 3):
                if self.bo[i][j] == num and (i,j) != pos:
                    return False

        return True


    def print_board(self):
        for i in range(len(self.bo)):
            if i % 3 == 0 and i != 0:
                print("- - - - - - - - - - - - - ")

            for j in range(len(self.bo[0])):
                if j % 3 == 0 and j != 0:
                    print(" | ", end="")

                if j == 8:
                    print(self.bo[i][j])
                else:
                    print(str(self.bo[i][j]) + " ", end="")


    def find_empty(self):
        for i in range(len(self.bo)):
            for j in range(len(self.bo[0])):
                if self.bo[i][j] == 0:
                    return (i, j)  # row, col

        return None
